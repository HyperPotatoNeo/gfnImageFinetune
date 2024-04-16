# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Extension of diffusers.StableDiffusionPipeline."""

from typing import Any, Callable, Dict, List, Optional, Union
from diffusers import StableDiffusionPipeline
import torch


class StableDiffusionPipelineExtended(StableDiffusionPipeline):
  """Extension of diffusers.StableDiffusionPipeline."""

  # Return the full trajectory
  def forward_collect_traj_ddim(
      self,
      prompt = None,
      height = None,
      width = None,
      num_inference_steps = 50,
      guidance_scale = 7.5,
      negative_prompt = None,
      num_images_per_prompt = 1,
      eta = 1.0,
      generator = None,
      latents = None,
      prompt_embeds = None,
      negative_prompt_embeds = None,
      output_type = None,
      return_dict = True,
      callback = None,
      callback_steps = 1,
      cross_attention_kwargs = None,
      is_ddp = False,
      unet_copy=None,
      soft_reward=False,
      gfn=False,
  ):
    # pylint: disable=line-too-long
    r"""Function invoked when calling the pipeline for generation.

    Args:
        prompt (`str` or `List[str]`, *optional*): The prompt or prompts to
          guide the image generation. If not defined, one has to pass
          `prompt_embeds` instead.
        height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
          The height in pixels of the generated image.
        width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
          The width in pixels of the generated image.
        num_inference_steps (`int`, *optional*, defaults to 50): The number of
          denoising steps. More denoising steps usually lead to a higher quality
          image at the expense of slower inference.
        guidance_scale (`float`, *optional*, defaults to 7.5): Guidance scale as
          defined in [Classifier-Free Diffusion
          Guidance](https://arxiv.org/abs/2207.12598). `guidance_scale` is
          defined as `w` of equation 2. of [Imagen
          Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is
          enabled by setting `guidance_scale > 1`. Higher guidance scale
          encourages to generate images that are closely linked to the text
          `prompt`, usually at the expense of lower image quality.
        negative_prompt (`str` or `List[str]`, *optional*): The prompt or
          prompts not to guide the image generation. If not defined, one has to
          pass `negative_prompt_embeds`. instead. If not defined, one has to
          pass `negative_prompt_embeds`. instead. Ignored when not using
          guidance (i.e., ignored if `guidance_scale` is less than `1`).
        num_images_per_prompt (`int`, *optional*, defaults to 1): The number of
          images to generate per prompt.
        eta (`float`, *optional*, defaults to 0.0): Corresponds to parameter eta
          (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies
          to [`schedulers.DDIMScheduler`], will be ignored for others.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
          One or a list of [torch
          generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
          to make generation deterministic.
        latents (`torch.FloatTensor`, *optional*): Pre-generated noisy latents,
          sampled from a Gaussian distribution, to be used as inputs for image
          generation. Can be used to tweak the same generation with different
          prompts. If not provided, a latents tensor will ge generated by
          sampling using the supplied random `generator`.
        prompt_embeds (`torch.FloatTensor`, *optional*): Pre-generated text
          embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
          weighting. If not provided, text embeddings will be generated from
          `prompt` input argument.
        negative_prompt_embeds (`torch.FloatTensor`, *optional*): Pre-generated
          negative text embeddings. Can be used to easily tweak text inputs,
          *e.g.* prompt weighting. If not provided, negative_prompt_embeds will
          be generated from `negative_prompt` input argument.
        output_type (`str`, *optional*, defaults to `"pil"`): The output format
          of the generate image. Choose between
          [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or
          `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`): Whether or not to
          return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`]
          instead of a plain tuple.
        callback (`Callable`, *optional*): A function that will be called every
          `callback_steps` steps during inference. The function will be called
          with the following arguments:
          `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
        callback_steps (`int`, *optional*, defaults to 1): The frequency at
          which the `callback` function will be called. If not specified, the
          callback will be called at every step.
        cross_attention_kwargs (`dict`, *optional*): A kwargs dictionary that if
          specified is passed along to the `AttnProcessor` as defined under
          `self.processor` in
          [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        is_ddp (`bool`, *optional*, defaults to `False`): whether the unet is a
          `DistributedDataParallel` model. If `True`, the `height` and `width`
          arguments will be calculated using the unwrapped model.
        unet_copy (`torch.nn.Module`, *optional*, defaults to `None`): the
          pretrained model to calculate soft reward
        soft_reward (`bool`, *optional*, defaults to `False`): whether to use
          soft reward

    Examples:

    Returns:
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or
        `tuple`:
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if
        `return_dict` is True, otherwise a `tuple.
        When returning a tuple, the first element is a list with the generated
        images, and the second element is a
        list of `bool`s denoting whether the corresponding generated image
        likely represents "not-safe-for-work"
        (nsfw) content, according to the `safety_checker`.
        latents_list (`List[torch.FloatTensor]`): A list of latents states
        unconditional_prompt_embeds (`List[torch.FloatTensor]`) A list of
        unconditional prompt embeddings
        guided_prompt_embeds (`List[torch.FloatTensor]`) A list of conditional
        prompt embeddings
        log_probs_list (`List[torch.FloatTensor]`): A list of log probabilities
        for each step
        kl_path_list (`List[torch.FloatTensor]`): A list of soft rewards for
        each step
    """
    # 0. Default height and width to unet
    if is_ddp:
      height = (
          height or self.unet.module.config.sample_size * self.vae_scale_factor
      )
      width = (
          width or self.unet.module.config.sample_size * self.vae_scale_factor
      )
    else:
      height = height or self.unet.config.sample_size * self.vae_scale_factor
      width = width or self.unet.config.sample_size * self.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
    )

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
      batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
      batch_size = len(prompt)
    else:
      batch_size = prompt_embeds.shape[0]

    device = self._execution_device
    # here `guidance_scale` is defined analog to the guidance weight `w` of
    # equation (2) of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf .
    # `guidance_scale = 1` corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    prompt_embeds = self._encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
    )

    # 4. Prepare timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps

    # 5. Prepare latent variables
    if is_ddp:
      num_channels_latents = self.unet.module.in_channels
    else:
      num_channels_latents = self.unet.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
    latents_list = []
    log_prob_list = []
    if gfn:
      old_log_prob_list = []
    latents_list.append(latents.detach().clone().cpu())
    # 6. Prepare extra step kwargs.
    # TODO: Logic should ideally just be moved out of the pipeline  # pylint: disable=g-bad-todo
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 7. Denoising loop
    num_warmup_steps = (
        len(timesteps) - num_inference_steps * self.scheduler.order
    )
    kl_list = []
    with self.progress_bar(total=num_inference_steps) as progress_bar:
      for i, t in enumerate(timesteps):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = (
            torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        )

        # predict the noise residual
        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
        ).sample

        if soft_reward or gfn:
          old_noise_pred = unet_copy(
              latent_model_input,
              t,
              encoder_hidden_states=prompt_embeds,
              cross_attention_kwargs=cross_attention_kwargs,
          ).sample

          if do_classifier_free_guidance:
            old_noise_pred_uncond, old_noise_pred_text = old_noise_pred.chunk(2)
            old_noise_pred = old_noise_pred_uncond + guidance_scale * (
                old_noise_pred_text - old_noise_pred_uncond
            )

        # perform guidance
        if do_classifier_free_guidance:
          noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
          noise_pred = noise_pred_uncond + guidance_scale * (
              noise_pred_text - noise_pred_uncond
          )

        # now we get the predicted noise
        unsqueeze3x = lambda x: x[Ellipsis, None, None, None]
        unet_t = unsqueeze3x(torch.tensor([t])).to(noise_pred.device)
        # compute the previous noisy sample x_t -> x_t-1
        if soft_reward or gfn:
          prev_latents = latents.clone()

        latents, log_prob = self.scheduler.step_logprob(
            noise_pred, unet_t, latents, **extra_step_kwargs
        )
        latents = latents.prev_sample
        latents = latents.to(prompt_embeds.dtype)
        latents_list.append(latents.detach().clone().cpu())
        log_prob_list.append(log_prob.detach().clone().cpu())

        if soft_reward or gfn:
          unet_times = unet_t
          old_log_prob = self.scheduler.step_forward_logprob(
          old_noise_pred, unet_times, prev_latents, latents, **extra_step_kwargs
          )

          #kl_list.append((log_prob - old_log_prob).detach().clone().cpu())
          old_log_prob_list.append(old_log_prob.detach().clone().cpu())

        # call the callback, if provided
        if i == len(timesteps) - 1 or (
            (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
        ):
          progress_bar.update()
          if callback is not None and i % callback_steps == 0:
            callback(i, t, latents)

    if output_type == "latent":
      image = latents
    elif output_type == "pil":
      # 8. Post-processing
      latents = latents.detach()
      latents = latents.to(prompt_embeds.dtype)
      image = self.decode_latents(latents)

      # 10. Convert to PIL
      image = self.numpy_to_pil(image)
    else:
      # 8. Post-processing
      image = self.decode_latents(latents)

    # Offload last model to CPU
    if (
        hasattr(self, "final_offload_hook")
        and self.final_offload_hook is not None
    ):
      self.final_offload_hook.offload()

    unconditional_prompt_embeds, guided_prompt_embeds = prompt_embeds.chunk(2)

    if soft_reward:
      kl_sum = 0
      kl_path = []
      for i in range(len(kl_list)):
        kl_sum += kl_list[i]
      kl_path.append(kl_sum.clone())
      for i in range(1, len(kl_list)):
        kl_sum -= kl_list[i - 1]
        kl_path.append(kl_sum.clone())
    else:
      kl_path = None

    if gfn:
      return (
        image,
        latents_list,
        unconditional_prompt_embeds.detach().cpu(),
        guided_prompt_embeds.detach().cpu(),
        log_prob_list,
        old_log_prob_list,
        kl_path,
    )
    return (
        image,
        latents_list,
        unconditional_prompt_embeds.detach().cpu(),
        guided_prompt_embeds.detach().cpu(),
        log_prob_list,
        kl_path,
    )

  # Feed transitions pairs and old model
  def forward_calculate_logprob(
      self,
      prompt_embeds,
      latents,
      next_latents,
      ts,
      unet_copy=None,
      height = None,
      width = None,
      num_inference_steps = 50,
      guidance_scale = 7.5,
      negative_prompt = None,
      num_images_per_prompt = 1,
      eta = 1.0,
      generator = None,
      negative_prompt_embeds = None,
      output_type = "pil",
      return_dict = True,
      callback = None,
      callback_steps = 1,
      cross_attention_kwargs = None,
      is_ddp = False,
      soft_reward=False,
  ):
    # pylint: disable=line-too-long
    r"""Function invoked when calling the pipeline for generation.

    Args:
        prompt_embeds (`torch.FloatTensor` of shape [batch_size, embed_dim]):
          prompt embeddings used in data generation.
        latents: current state.
        next_latents: next state.
        ts (`torch.LongTensor`): timesteps to calculate logprob.
        unet_copy (`torch.nn.Module`, *optional*, defaults to `None`): the
          pretrained model to calculate soft reward.
        height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
          The height in pixels of the generated image.
        width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
          The width in pixels of the generated image.
        num_inference_steps (`int`, *optional*, defaults to 50): The number of
          denoising steps. More denoising steps usually lead to a higher quality
          image at the expense of slower inference.
        guidance_scale (`float`, *optional*, defaults to 7.5): Guidance scale as
          defined in [Classifier-Free Diffusion
          Guidance](https://arxiv.org/abs/2207.12598). `guidance_scale` is
          defined as `w` of equation 2. of [Imagen
          Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is
          enabled by setting `guidance_scale > 1`. Higher guidance scale
          encourages to generate images that are closely linked to the text
          `prompt`, usually at the expense of lower image quality.
        negative_prompt (`str` or `List[str]`, *optional*): The prompt or
          prompts not to guide the image generation. If not defined, one has to
          pass `negative_prompt_embeds`. instead. If not defined, one has to
          pass `negative_prompt_embeds`. instead. Ignored when not using
          guidance (i.e., ignored if `guidance_scale` is less than `1`).
        num_images_per_prompt (`int`, *optional*, defaults to 1): The number of
          images to generate per prompt.
        eta (`float`, *optional*, defaults to 0.0): Corresponds to parameter eta
          (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies
          to [`schedulers.DDIMScheduler`], will be ignored for others.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
          One or a list of [torch
          generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
          to make generation deterministic.
        negative_prompt_embeds (`torch.FloatTensor`, *optional*): Pre-generated
          negative text embeddings. Can be used to easily tweak text inputs,
          *e.g.* prompt weighting. If not provided, negative_prompt_embeds will
          be generated from `negative_prompt` input argument.
        output_type (`str`, *optional*, defaults to `"pil"`): The output format
          of the generate image. Choose between
          [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or
          `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`): Whether or not to
          return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`]
          instead of a plain tuple.
        callback (`Callable`, *optional*): A function that will be called every
          `callback_steps` steps during inference. The function will be called
          with the following arguments:
          `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
        callback_steps (`int`, *optional*, defaults to 1): The frequency at
          which the `callback` function will be called. If not specified, the
          callback will be called at every step.
        cross_attention_kwargs (`dict`, *optional*): A kwargs dictionary that if
          specified is passed along to the `AttnProcessor` as defined under
          `self.processor` in
          [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        is_ddp (`bool`, *optional*, defaults to `False`): whether the unet is a
          `DistributedDataParallel` model. If `True`, the `height` and `width`
          arguments will be calculated using the unwrapped model.
        soft_reward (`bool`, *optional*, defaults to `False`): whether to use
          soft reward
    Examples:

    Returns:
        log probability and KL regularizer.
    """
    if soft_reward:
      unet_copy = None
    # 0. Default height and width to unet
    if is_ddp:
      height = (
          height or self.unet.module.config.sample_size * self.vae_scale_factor
      )
      width = (
          width or self.unet.module.config.sample_size * self.vae_scale_factor
      )
    else:
      height = height or self.unet.config.sample_size * self.vae_scale_factor
      width = width or self.unet.config.sample_size * self.vae_scale_factor

    # 2. Define call parameters
    device = self._execution_device
    # here `guidance_scale` is defined analog to the guidance weight `w` of
    # equation (2) of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf .
    # `guidance_scale = 1` corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Prepare timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps
    unet_times = timesteps[ts]

    # 4. Prepare latent variables
    latents_list = []
    latents_list.append(latents.detach().clone())
    # 5. Prepare extra step kwargs.
    # TODO: Logic should ideally just be moved out of the pipeline  # pylint: disable=g-bad-todo
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 6. Denoising loop
    # for training loops:
    # with self.progress_bar(total=num_inference_steps) as progress_bar:
    # for i, t in enumerate(timesteps):
    # expand the latents if we are doing classifier free guidance
    latent_model_input = (
        torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    )
    time_model_input = (
        torch.cat([unet_times] * 2)
        if do_classifier_free_guidance
        else unet_times
    )

    # predict the noise residual
    noise_pred = self.unet(
        latent_model_input,
        time_model_input,
        encoder_hidden_states=prompt_embeds,
        cross_attention_kwargs=cross_attention_kwargs,
    ).sample

    # add regularization

    if unet_copy is not None:
      old_noise_pred = unet_copy(
          latent_model_input,
          time_model_input,
          encoder_hidden_states=prompt_embeds,
          cross_attention_kwargs=cross_attention_kwargs,
      ).sample

    # perform guidance
    if do_classifier_free_guidance:
      noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
      noise_pred = noise_pred_uncond + guidance_scale * (
          noise_pred_text - noise_pred_uncond
      )

    if unet_copy is not None:
      if do_classifier_free_guidance:
        old_noise_pred_uncond, old_noise_pred_text = old_noise_pred.chunk(2)
        old_noise_pred = old_noise_pred_uncond + guidance_scale * (
            old_noise_pred_text - old_noise_pred_uncond
        )
        # now we get the predicted noise
      kl_regularizer = (noise_pred - old_noise_pred) ** 2
    else:
      kl_regularizer = torch.zeros(noise_pred.shape[0])

    unsqueeze3x = lambda x: x[Ellipsis, None, None, None]
    unet_times = unsqueeze3x(unet_times).to(noise_pred.device)
    log_prob = self.scheduler.step_forward_logprob(
        noise_pred, unet_times, latents, next_latents, **extra_step_kwargs
    )

    return log_prob, kl_regularizer
