#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
import os, random, time, json
from random import randint
from lpipsPyTorch import lpips
from utils.loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim
from gaussian_renderer import render_fastgs, network_gui_ws
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras

# 训练配置
LARGE_GAUSSIAN_CONFIG = {
    "prune_before_eval": False,   # 评估前过滤
    "prune_before_save": False,   # 保存前过滤
    "scale_multiplier": 10.0,     # 过滤尺寸 > 中位数 × N倍 的高斯
    "eval_image_count": 5,        # 评估时从训练集选取的影像数量
    "eval_select_mode": "uniform", # 选取模式: uniform, middle, random, manual
    "eval_image_indices": [],      # 手动指定的影像索引（manual模式）
    "eval_sort_by_name": False,    # 是否按文件名升序排序
}

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, websockets):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Dictionary to store metrics during training
    metrics_dict = {}
    
    # 固定用于评估的相机（在训练开始时选择，之后保持不变）
    all_train_cameras = scene.getTrainCameras()
    all_test_cameras = scene.getTestCameras()
    
    # 如果启用按文件名排序，则对相机列表排序
    if LARGE_GAUSSIAN_CONFIG.get("eval_sort_by_name", False):
        all_train_cameras = sorted(all_train_cameras, key=lambda x: x.image_name)
        print("Camera list sorted by name (ascending)")
    
    # 从训练集中选择指定数量的相机用于评估
    eval_count = LARGE_GAUSSIAN_CONFIG.get("eval_image_count", 5)
    eval_mode = LARGE_GAUSSIAN_CONFIG.get("eval_select_mode", "uniform")
    n_total = len(all_train_cameras)
    
    if eval_mode == "manual":
        # 手动指定索引
        manual_indices = LARGE_GAUSSIAN_CONFIG.get("eval_image_indices", [])
        # 过滤掉超出范围的索引
        train_eval_indices = [i for i in manual_indices if 0 <= i < n_total]
        if len(train_eval_indices) == 0:
            print(f"Warning: no valid manual indices, falling back to uniform")
            train_eval_indices = [i * n_total // eval_count for i in range(min(eval_count, n_total))]
    elif n_total <= eval_count:
        train_eval_indices = list(range(n_total))
    elif eval_mode == "uniform":
        # 均匀分散选取（从头到尾均匀间隔）
        # 例: 100张选3张 -> 索引 0, 33, 66
        train_eval_indices = [i * n_total // eval_count for i in range(eval_count)]
    elif eval_mode == "middle":
        # 中间部分均匀选取（避开首尾 10%）
        # 例: 100张选3张 -> 从10-90范围内均匀选取
        start = n_total // 10
        end = n_total - n_total // 10
        middle_range = end - start
        train_eval_indices = [start + i * middle_range // eval_count for i in range(eval_count)]
    elif eval_mode == "random":
        # 随机选取（使用固定种子保证可重复）
        import random as rand_module
        rand_module.seed(42)
        train_eval_indices = sorted(rand_module.sample(range(n_total), eval_count))
    else:
        # 默认均匀分散
        train_eval_indices = [i * n_total // eval_count for i in range(eval_count)]
    
    fixed_train_eval_cameras = [all_train_cameras[i] for i in train_eval_indices]
    # 打印选中的相机名称
    eval_names = [fixed_train_eval_cameras[i].image_name for i in range(len(fixed_train_eval_cameras))]
    print(f"Eval camera indices ({eval_mode}): {train_eval_indices}")
    print(f"Eval camera names: {eval_names}")
    
    # 打印所有相机列表（帮助确认索引对应关系）
    print(f"\n=== All train cameras ({n_total} total) ===")
    for i, cam in enumerate(all_train_cameras):
        marker = " <-- SELECTED" if i in train_eval_indices else ""
        print(f"  [{i}] {cam.image_name}{marker}")
    print("=" * 40 + "\n")
    # 测试集保持不变（如果有的话）
    fixed_test_eval_cameras = all_test_cameras if all_test_cameras else []
    print(f"Fixed evaluation cameras: {len(fixed_train_eval_cameras)} train, {len(fixed_test_eval_cameras)} test")
    
    # 打印初始高斯统计信息（用于参考）
    if LARGE_GAUSSIAN_CONFIG["prune_before_eval"] or LARGE_GAUSSIAN_CONFIG["prune_before_save"]:
        init_scales = gaussians.get_scaling.max(dim=1).values
        print(f"\n=== Initial Gaussian Scale Distribution ===")
        print(f"Total gaussians: {init_scales.shape[0]}")
        print(f"Scale: min={init_scales.min():.6f}, median={init_scales.median():.6f}, mean={init_scales.mean():.6f}, max={init_scales.max():.6f}")
        print(f"(Note: scales will decrease during training)")
        print(f"============================================\n")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    # record time
    optim_start = torch.cuda.Event(enable_timing=True)
    optim_end = torch.cuda.Event(enable_timing=True)
    total_time = 0.0

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    bg = torch.rand((3), device="cuda") if opt.random_background else background

    for iteration in range(first_iter, opt.iterations + 1):

        if websockets:
            if network_gui_ws.curr_id >= 0 and network_gui_ws.curr_id < len(scene.getTrainCameras()):
                cam = scene.getTrainCameras()[network_gui_ws.curr_id]
                net_image = render_fastgs(cam, gaussians, pipe, background, opt.mult, 1.0)["render"]
                network_gui_ws.latest_width = cam.image_width
                network_gui_ws.latest_height = cam.image_height
                network_gui_ws.latest_result = net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())

        iter_start.record()
        
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        _ = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render_fastgs(viewpoint_cam, gaussians, pipe, bg, opt.mult)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            iter_time = iter_start.elapsed_time(iter_end)
            
            # 评估前过滤超大高斯（让指标更准确）
            if LARGE_GAUSSIAN_CONFIG["prune_before_eval"] and iteration in testing_iterations:
                n_before = gaussians.get_xyz.shape[0]
                max_scales = gaussians.get_scaling.max(dim=1).values
                median_scale = max_scales.median()
                mean_scale = max_scales.mean()
                max_scale = max_scales.max()
                threshold = median_scale * LARGE_GAUSSIAN_CONFIG["scale_multiplier"]
                prune_mask = max_scales > threshold
                n_prune = prune_mask.sum().item()
                # 总是打印统计信息，帮助调试
                print(f"\n[ITER {iteration}] Gaussians before filter: {n_before}")
                print(f"[ITER {iteration}] Scale stats: median={median_scale:.6f}, mean={mean_scale:.6f}, max={max_scale:.6f}")
                print(f"[ITER {iteration}] Threshold={threshold:.6f}, to_prune={n_prune}")
                if n_prune > 0:
                    gaussians.prune_points(prune_mask)
                    n_after = gaussians.get_xyz.shape[0]
                    # 验证 scene.gaussians 是否也更新了
                    n_scene = scene.gaussians.get_xyz.shape[0]
                    print(f"[ITER {iteration}] Pre-eval pruning: {n_before} -> {n_after} (scene.gaussians: {n_scene})")
                    # 再次检查 max scale
                    new_max = gaussians.get_scaling.max(dim=1).values.max()
                    print(f"[ITER {iteration}] New max scale after filter: {new_max:.6f}")
            
            # Log and save（使用固定的评估相机）
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_time, testing_iterations, scene, render_fastgs, (pipe, background, opt.mult), dataset.model_path, metrics_dict, fixed_train_eval_cameras, fixed_test_eval_cameras)
            if (iteration in saving_iterations):
                # 保存前过滤（如果评估时已过滤则跳过）
                already_pruned = LARGE_GAUSSIAN_CONFIG["prune_before_eval"] and iteration in testing_iterations
                if LARGE_GAUSSIAN_CONFIG["prune_before_save"] and not already_pruned:
                    max_scales = gaussians.get_scaling.max(dim=1).values
                    median_scale = max_scales.median()
                    threshold = median_scale * LARGE_GAUSSIAN_CONFIG["scale_multiplier"]
                    prune_mask = max_scales > threshold
                    n_prune = prune_mask.sum().item()
                    if n_prune > 0:
                        gaussians.prune_points(prune_mask)
                        print(f"\n[ITER {iteration}] Pre-save pruning: removed {n_prune} large gaussians (>{LARGE_GAUSSIAN_CONFIG['scale_multiplier']}x median)")
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            optim_start.record()
            
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    my_viewpoint_stack = scene.getTrainCameras().copy()
                    camlist = sampling_cameras(my_viewpoint_stack)

                    # The multiview consistent densification of fastgs
                    importance_score, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt, DENSIFY=True)                    
                    gaussians.densify_and_prune_fastgs(max_screen_size = size_threshold, 
                                                min_opacity = 0.005, 
                                                extent = scene.cameras_extent, 
                                                radii=radii,
                                                args = opt,
                                                importance_score = importance_score,
                                                pruning_score = pruning_score)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()


            # The multiview consistent pruning of fastgs. We do it every 3k iterations after 15k
            # In this stage, the model converge basically. So we can prune more aggressively without degrading rendering quality.
            # You can check the rendering results of 20K iterations in arxiv version (https://arxiv.org/abs/2511.04283), the rendering quality is already very good.
            if iteration % 3000 == 0 and iteration > 15_000 and iteration < 30_000:
                my_viewpoint_stack = scene.getTrainCameras().copy()
                camlist = sampling_cameras(my_viewpoint_stack)

                _, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt)                    
                gaussians.final_prune_fastgs(min_opacity = 0.1, pruning_score = pruning_score)
        
            # Optimization step
            if iteration < opt.iterations:
                if opt.optimizer_type == "default":
                    gaussians.optimizer_step(iteration)
                elif opt.optimizer_type == "sparse_adam":
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)

            # record time
            optim_end.record()
            torch.cuda.synchronize()
            optim_time = optim_start.elapsed_time(optim_end)
            total_time += (iter_time + optim_time) / 1e3

    # scene.save(iteration)
    print(f"Gaussian number: {gaussians._xyz.shape[0]}")
    print(f"Training time: {total_time}")
    
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, model_path=None, metrics_dict=None, fixed_train_cameras=None, fixed_test_cameras=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        # 使用固定的评估相机（如果提供），否则使用默认逻辑
        if fixed_train_cameras is not None:
            train_cameras = fixed_train_cameras
        else:
            train_cameras = [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]
        if fixed_test_cameras is not None:
            test_cameras = fixed_test_cameras
        else:
            test_cameras = scene.getTestCameras()
        validation_configs = ({'name': 'test', 'cameras' : test_cameras}, 
                              {'name': 'train', 'cameras' : train_cameras})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0
                
                print(f"\n[ITER {iteration}] Evaluating {config['name']} ({len(config['cameras'])} images):")
                print("-" * 70)
                
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    # 计算当前影像的指标
                    img_l1 = l1_loss(image, gt_image).mean().double()
                    img_psnr = psnr(image, gt_image).mean().double()
                    img_ssim = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double()
                    img_lpips = lpips(image, gt_image, net_type='vgg').mean().double()
                    
                    # 打印每张图片的指标
                    print(f"  [{idx}] {viewpoint.image_name}: PSNR={img_psnr:.2f}, SSIM={img_ssim:.4f}, LPIPS={img_lpips:.4f}")
                    
                    # 累加用于计算平均值
                    l1_test += img_l1
                    psnr_test += img_psnr
                    ssim_test += img_ssim
                    lpips_test += img_lpips
                    
                    # 记录每个影像的指标和渲染图到 TensorBoard
                    if tb_writer:
                        tag_prefix = f"{config['name']}_view_{viewpoint.image_name}"
                        # 记录每个影像的单独指标
                        tb_writer.add_scalar(f"{tag_prefix}/PSNR", img_psnr, iteration)
                        tb_writer.add_scalar(f"{tag_prefix}/SSIM", img_ssim, iteration)
                        tb_writer.add_scalar(f"{tag_prefix}/LPIPS", img_lpips, iteration)
                        # 记录渲染图
                        tb_writer.add_images(f"{tag_prefix}/render_step_{iteration}", image[None], global_step=iteration)
                        # ground_truth 只在第一个测试点记录一次
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f"{tag_prefix}/ground_truth", gt_image[None], global_step=iteration)
                
                # 计算平均值
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                
                # 打印平均指标
                print("-" * 70)
                print(f"  [AVG] {config['name']}: PSNR={psnr_test:.2f}, SSIM={ssim_test:.4f}, LPIPS={lpips_test:.4f}, L1={l1_test:.4f}")
                
                # Save metrics to dictionary
                if metrics_dict is not None:
                    if config['name'] not in metrics_dict:
                        metrics_dict[config['name']] = {}
                    metrics_dict[config['name']][iteration] = {
                        'L1': float(l1_test),
                        'PSNR': float(psnr_test),
                        'SSIM': float(ssim_test),
                        'LPIPS': float(lpips_test),
                        'num_gaussians': scene.gaussians.get_xyz.shape[0]
                    }
                    # Save to JSON file
                    if model_path:
                        metrics_path = os.path.join(model_path, "training_metrics.json")
                        with open(metrics_path, 'w') as f:
                            json.dump(metrics_dict, f, indent=2)
                
                if tb_writer:
                    # 记录平均指标到 TensorBoard
                    tb_writer.add_scalar(f"{config['name']}_avg/PSNR", psnr_test, iteration)
                    tb_writer.add_scalar(f"{config['name']}_avg/SSIM", ssim_test, iteration)
                    tb_writer.add_scalar(f"{config['name']}_avg/LPIPS", lpips_test, iteration)
                    tb_writer.add_scalar(f"{config['name']}_avg/L1", l1_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # ============ 调试配置 ============
    # 输入路径：COLMAP 数据目录（包含 images/ 和 sparse/ 子目录）
    DEBUG_SOURCE_PATH = r"D:\Github_code\FastGS\data\Ganluo_Test"
    # 输出路径：训练结果保存目录
    DEBUG_MODEL_PATH = r"D:\Github_code\FastGS\output\test"
    # 最大迭代次数（调试时设小一点，正式训练用 30000）
    DEBUG_ITERATIONS = 10000
    # 测试迭代点：在这些迭代次数时进行评估（计算 PSNR 等指标）
    DEBUG_TEST_ITERATIONS = [1000, 3000, 6000, 10000, 15000, 20000]
    # 保存迭代点：在这些迭代次数时保存模型
    DEBUG_SAVE_ITERATIONS = [1000, 3000, 6000, 10000, 15000, 20000]
    # 是否禁用点的增加（Densification）
    # True = 禁用，只使用初始点云训练；False = 启用，训练过程中会增加点
    DISABLE_DENSIFICATION = True
    # 是否在评估指标前过滤超大高斯（让 PSNR/SSIM 更准确）
    PRUNE_BEFORE_EVAL = True
    # 是否在保存点云前过滤超大高斯（保存的模型更干净）
    PRUNE_BEFORE_SAVE = True
    # 过滤倍数：尺寸 > 中位数 × N倍 的高斯会被过滤
    # 比如设置 3.0，表示尺寸超过中位数 3 倍的高斯会被移除
    SCALE_MULTIPLIER = 5.0
    # 评估时从训练集中选取的影像数量（仅 uniform/middle/random 模式有效）
    EVAL_IMAGE_COUNT = 5
    # 评估影像选取模式:
    #   "uniform"  - 均匀分散选取（从头到尾均匀间隔，使用 EVAL_IMAGE_COUNT）
    #   "middle"   - 中间部分均匀选取（避开首尾，使用 EVAL_IMAGE_COUNT）
    #   "random"   - 随机选取（训练开始时固定，使用 EVAL_IMAGE_COUNT）
    #   "manual"   - 手动指定（使用 EVAL_IMAGE_INDICES，忽略 EVAL_IMAGE_COUNT）
    EVAL_SELECT_MODE = "manual"
    # 手动指定评估影像的索引（仅当 EVAL_SELECT_MODE = "manual" 时有效）
    # 例如 [0, 10, 20] 表示选取第 0、10、20 张图片（此时 EVAL_IMAGE_COUNT 无效）
    # EVAL_IMAGE_INDICES = [2, 4, 6, 8, 10]
    EVAL_IMAGE_INDICES = [4, 8, 12, 16, 20] #34张Ganluo
    # 是否按文件名升序排序相机列表（默认是 COLMAP 内部顺序）
    EVAL_SORT_BY_NAME = True
    # ==================================
    
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[50_000, 100_000, 150_000, 200_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[50_000, 100_000, 150_000, 200_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[50_000, 100_000, 150_000, 200_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--websockets", action='store_true', default=False)
    parser.add_argument("--benchmark_dir", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    
    # ============ 应用调试配置 ============
    # 如果命令行没有指定，则使用调试默认值
    if not args.source_path:
        args.source_path = DEBUG_SOURCE_PATH
    if not args.model_path:
        args.model_path = DEBUG_MODEL_PATH
    if args.iterations == 30_000:  # 默认值未被修改时
        args.iterations = DEBUG_ITERATIONS
        # 调整保存/测试迭代点以匹配较小的总迭代次数
        args.test_iterations = DEBUG_TEST_ITERATIONS
        args.save_iterations = DEBUG_SAVE_ITERATIONS
        args.checkpoint_iterations = DEBUG_SAVE_ITERATIONS  # checkpoint 与 save 保持一致
    # 禁用 Densification（不生成额外点）
    if DISABLE_DENSIFICATION:
        args.densify_until_iter = 0
        print("Densification disabled: using only initial point cloud")
    # 设置超大高斯过滤参数
    LARGE_GAUSSIAN_CONFIG["prune_before_eval"] = PRUNE_BEFORE_EVAL
    LARGE_GAUSSIAN_CONFIG["prune_before_save"] = PRUNE_BEFORE_SAVE
    LARGE_GAUSSIAN_CONFIG["scale_multiplier"] = SCALE_MULTIPLIER
    LARGE_GAUSSIAN_CONFIG["eval_image_count"] = EVAL_IMAGE_COUNT
    LARGE_GAUSSIAN_CONFIG["eval_select_mode"] = EVAL_SELECT_MODE
    LARGE_GAUSSIAN_CONFIG["eval_image_indices"] = EVAL_IMAGE_INDICES
    LARGE_GAUSSIAN_CONFIG["eval_sort_by_name"] = EVAL_SORT_BY_NAME
    if PRUNE_BEFORE_EVAL or PRUNE_BEFORE_SAVE:
        print(f"Large gaussian pruning: scale > {SCALE_MULTIPLIER}x median, before_eval={PRUNE_BEFORE_EVAL}, before_save={PRUNE_BEFORE_SAVE}")
    if EVAL_SELECT_MODE == "manual":
        print(f"Evaluation images: manual indices {EVAL_IMAGE_INDICES}")
    else:
        print(f"Evaluation images: {EVAL_IMAGE_COUNT} ({EVAL_SELECT_MODE} mode)")
    # =====================================
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if(args.websockets):
        network_gui_ws.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(
        lp.extract(args), 
        op.extract(args), 
        pp.extract(args), 
        args.test_iterations, 
        args.save_iterations, 
        args.checkpoint_iterations, 
        args.start_checkpoint, 
        args.debug_from, 
        args.websockets
    )

    # All done
    print("\nTraining complete.")
