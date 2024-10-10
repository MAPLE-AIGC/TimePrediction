# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import os
import time
import json
import psutil
import numpy as np
import torch
from diffusers import AutoencoderKL
import torch.distributed

import dnnlib
import copy
from torch.optim.lr_scheduler import LambdaLR
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
from training.utils.loggin import DiffusionLossLogger
from torchvision.utils import make_grid


@torch.no_grad()
def vae_decode(vae, latents, dtype):
    image = vae.decode(latents.to(dtype) / vae.config.scaling_factor, return_dict=False)[0] 
    image = (image / 2 + 0.5).clamp(0, 1)
    
    return image

#----------------------------------------------------------------------------

@torch.inference_mode()
def validation_step(
    net, 
    prompt,
    dtype,
    seed: int = 112,
):

    torch.manual_seed(seed)
    device = 'cuda'
    bs = len(prompt)

    text_embeddings, uncond_embeddings = net.encode_prompt(prompt)
    
    latents = torch.randn(bs, 4, 64, 64, device=device, dtype=dtype)    
    latents = latents * net.scheduler.init_noise_sigma

    variance_noises = [torch.randn_like(latents) for _ in range(1000)]
    epsilon = torch.tensor([5], device=device)
    t_cur = torch.tensor([999], device=device)

    while t_cur > epsilon:
        print(t_cur)
        noise_pred,next_timestep = net.generate(
            latents=latents,
            t=t_cur,
            prompt_embeddings=text_embeddings, 
            uncond_embeddings=uncond_embeddings,
            predict_next_timetsep=True,
            do_classifier_free_guidance=True


        )
        latents = net.step(
            t_cur,
            latents, 
            noise_pred, 
            next_timestep,
            enable_next_timesteps=True,
            validation = True,
            variance_noises = variance_noises
        )
        t_cur=next_timestep.to(t_cur.dtype)
    latents = latents / net.vae.config.scaling_factor
    image = net.vae.decode(latents, return_dict=False)[0]
    image = make_grid(image, nrow=4).float()
    image = (image *.5 + .5).clamp(0, 1)
        
    return image



#----------------------------------------------------------------------------

def training_loop(
    run_dir             = '.',      # Output directory.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    loss_kwargs         = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    augment_kwargs      = None,     # Options for augmentation pipeline, None = disable.
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    total_steps         = 200000,   # Training duration, measured in thousands of training images.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    step_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
    grad_accumulation   = 1,
    ema_kwargs          = {},
    lr_scheduler_kwargs = {},
    precision           = "fp16",
    resume_pt           = None,
    resume_state_dump   = None,
    resume_step         = 0,
    max_grad_norm       = 1000,
    vae_checkpoint      = '',
    val_denoising_step  = 20,
    val_ticks           = 5,
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    precision_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]

    num_accumulation_rounds = grad_accumulation

    # Load dataset.
    dist.print0('Loading dataset...')
    dataloader_iterator = dnnlib.util.construct_class_by_name(**data_loader_kwargs) # subclass of training.dataset.Dataset
    
    # Construct network.
    dist.print0('Constructing network...')
    net = dnnlib.util.construct_class_by_name(**network_kwargs) # subclass of torch.nn.Module
    net.to(device, dtype=precision_dtype)

    #net.train().requires_grad_(True).to(device, dtype=precision_dtype)
    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) # training.loss.(VP|VE|EDM)Loss
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
    ddp_net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device])
    ema = dnnlib.util.construct_class_by_name(model=net, **ema_kwargs)
    
    # Setup LR scheduler
    scheduler = LambdaLR(optimizer, lr_lambda=dnnlib.util.construct_class_by_name(**lr_scheduler_kwargs))

    # Resume training from previous snapshot.
    if resume_pt is not None and os.path.exists(resume_pt):
        dist.print0(f'Loading network weights from "{resume_pt}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        data = torch.load(resume_pt, map_location="cpu")
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        ema.load_state_dict(data['ema'])
        del data # conserve memory
        
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location="cpu")
        net.load_state_dict(data['net'])
        optimizer.load_state_dict(data['optimizer_state'])
        scheduler.load_state_dict(data['lr_scheduler'])
        del data # conserve memory

    dataloader_iterator = iter(dataloader_iterator)
    
    # Train.
    cur_tick = 0
    cur_nclip = 0
    training_step = resume_step # 0 for default
    tick_start_step = training_step
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    stats_jsonl = None
    
    
    is_flow = 'flow' in loss_kwargs['class_name'].lower()

    # tensorboard 
    if dist.get_rank() == 0:
        logger = DiffusionLossLogger(
            run_dir=run_dir,
            sigma_max=80.0 if not is_flow else 1.0,
            sigma_min=0.002,
            log_scale=not is_flow
        )
    #initial_params = {name: param.clone() for name, param in net.fc.named_parameters()}
    
    dist.print0()
    
    while True:
        if dist.get_rank() == 0 and not os.path.exists(run_dir): 
            assert False, f"run_dir {run_dir} does not exist"
        optimizer.zero_grad(set_to_none=True)
        losses = []
        sigmas = []
        # gradient accumulation
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp_net, (round_idx == num_accumulation_rounds - 1)):
                while True:
                    try:
                        batch = next(dataloader_iterator)
                        break
                    except Exception as e:
                        #print('reading next batch due to', e)
                        continue
                with torch.autocast(device_type="cuda", enabled=True, dtype=precision_dtype):
                    return_dict = loss_fn(net=ddp_net, batch=batch)
                #loss, sigmas= return_dict['loss'], return_dict['sigma']
                loss= return_dict['loss']
                training_stats.report('Loss/loss', loss)

                loss.backward()
            losses.append(loss.detach().mean(dim=[1, 2]))
        
        if dist.get_rank() == 0:
            logger.logger.add_scalar('Loss/loss', loss.detach().cpu().item(), global_step=training_step)
        
        # loss logging begin
        #sigmas = torch.concat(sigmas, dim=0)
        losses = torch.concat(losses, dim=0)
        #sigma_tensor_list = [torch.zeros_like(sigmas) for _ in range(dist.get_world_size())]
        loss_tensor_list = [torch.zeros_like(losses) for _ in range(dist.get_world_size())]
        # gather
        #torch.distributed.gather(
        #    sigmas, sigma_tensor_list if dist.get_rank() == 0 else None, dst=0)
        torch.distributed.gather(
            losses, loss_tensor_list if dist.get_rank() == 0 else None, dst=0)
        # if dist.get_rank() == 0:
        #     logger.record_loss(
        #         losses=torch.concat(loss_tensor_list, dim=0),
        #         sigmas=torch.concat(sigma_tensor_list, dim=0),
        #     )
        # loss logging end
        
        scheduler.step(training_step)
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=max_grad_norm)
        
        # Update weights.
        optimizer.step()

        # Update EMA.
        ema.update(net)

        # for name, param in net.fc.named_parameters():
        #     print("name",param)

            # if torch.equal(initial_params[name], param):
            #     print(f"{name} has not been updated.")
            # else:
            #     print(f"{name} has been updated.")
        cur_nclip += batch_size * num_accumulation_rounds
        done = (training_step >= total_steps)
        training_step += 1
        # Perform maintenance tasks once per tick.
        if (not done) and (cur_tick != 0) and (training_step < tick_start_step + step_per_tick):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"step {training_stats.report0('Progress/step', training_step)}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kclip {training_stats.report0('Timing/sec_per_kclip', (tick_end_time - tick_start_time) / cur_nclip * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0('\t'.join(fields))

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(
                ema=ema.state_dict(), 
                augment_pipe=augment_pipe
            )
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.get_rank() == 0:
                torch.save(data, os.path.join(run_dir, f'network-snapshot-{training_step:06d}.pt'))
            del data # conserve memory

        # Save full dump of the training state.
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            torch.save(dict(
                    net=net.state_dict(), 
                    optimizer_state=optimizer.state_dict(), 
                    lr_scheduler=scheduler.state_dict()
                ), 
                os.path.join(run_dir, f'training-state-{training_step:06d}.pt'))

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
            # tensorboard logger
            logger.log_diffusion_loss(global_step=training_step)

        # validation 
        if cur_tick % val_ticks == 0 and dist.get_rank() == 0:
            net.eval()
            with torch.autocast(device_type="cuda", enabled=True, dtype=precision_dtype):
                images = validation_step(
                    net=net, 
                    prompt=[
                        "a photograph of an astronaut riding a horse",
                    ],

                    dtype=precision_dtype,
                    seed=112,
                )
            images = images.cpu().detach().float()
            net.fc.train()
            from PIL import Image
            print(images.shape)
            image = images.permute(1, 2, 0)

            # 将张量转换为 numpy 数组，并将数值范围从 [0, 1] 或 [0, 255] 转换为 8-bit
            image = (image.numpy() * 255).astype(np.uint8)
            image_pil = Image.fromarray(image)

            # 保存图片s
            image_pil.save('output_image.png')
            logger.logger.add_image("Images/Generated", images, global_step=training_step)
        # Update state.
        cur_tick += 1
        cur_nclip = 0
        tick_start_step = training_step
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        # sync
        torch.distributed.barrier()
        if done:
            break

    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------
