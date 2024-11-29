import os, shutil, json, requests, random, time, runpod
from urllib.parse import urlsplit

import torch
from PIL import Image
import numpy as np

from nodes import NODE_CLASS_MAPPINGS, load_custom_node
from comfy_extras import  nodes_flux, nodes_differential_diffusion, nodes_model_advanced, nodes_custom_sampler

load_custom_node("/content/ComfyUI/custom_nodes/comfyui-art-venture")
load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI_LayerStyle")
load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI_essentials")
load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI-Advanced-ControlNet")
load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI_SLK_joy_caption_two")

CheckpointLoaderSimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
LoraLoader = NODE_CLASS_MAPPINGS["LoraLoader"]()
ACN_ControlNet = NODE_CLASS_MAPPINGS["ACN_ControlNet++LoaderSingle"]()
LoadBiRefNetModel = NODE_CLASS_MAPPINGS["LayerMask: LoadBiRefNetModel"]()

LoadImage = NODE_CLASS_MAPPINGS["LoadImage"]()
ImageBatch = NODE_CLASS_MAPPINGS["ImageBatch"]()
AV_IPAdapter = NODE_CLASS_MAPPINGS["AV_IPAdapter"]()

ImageScaleToMegapixels = NODE_CLASS_MAPPINGS["ImageScaleToMegapixels"]()
BiRefNetUltraV2 = NODE_CLASS_MAPPINGS["LayerMask: BiRefNetUltraV2"]()
ImageRemoveAlpha = NODE_CLASS_MAPPINGS["LayerUtility: ImageRemoveAlpha"]()
ImageDesaturate = NODE_CLASS_MAPPINGS["ImageDesaturate+"]()
AV_ControlNetPreprocessor = NODE_CLASS_MAPPINGS["AV_ControlNetPreprocessor"]()
ControlNetApplyAdvanced = NODE_CLASS_MAPPINGS["ControlNetApplyAdvanced"]()
CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
Joy_caption_two_load = NODE_CLASS_MAPPINGS["Joy_caption_two_load"]()
Joy_caption_two = NODE_CLASS_MAPPINGS["Joy_caption_two"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
GetImageSize = NODE_CLASS_MAPPINGS["GetImageSize+"]()
KSampler = NODE_CLASS_MAPPINGS["KSampler"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()

with torch.inference_mode():
    unet, clip, vae = CheckpointLoaderSimple.load_checkpoint("sdxl/leosamsHelloworldXL_helloworldXL70.safetensors")
    lora_unet, lora_clip = LoraLoader.load_lora(unet, clip, "sdxl/araminta_k_midsommar_cartoon.safetensors", 0.80, 1.0)
    control_net = ACN_ControlNet.load_controlnet_plusplus("sdxl/controlnet-union-sdxl-1.0-promax.safetensors", "canny/lineart/mlsd")[0]
    birefnet_model = LoadBiRefNetModel.load_birefnet_model("BiRefNet-general-epoch_244.pth")[0]
    joy_two_pipeline = Joy_caption_two_load.generate("Llama-3.1-8B-Lexi-Uncensored-V2")[0]

def download_file(url, save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)
    file_suffix = os.path.splitext(urlsplit(url).path)[1]
    file_name_with_suffix = file_name + file_suffix
    file_path = os.path.join(save_dir, file_name_with_suffix)
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

@torch.inference_mode()
def generate(input):
    values = input["input"]

    input_image = values['input_image']
    input_image = download_file(url=input_image, save_dir='/content/ComfyUI/input', file_name='input_image')
    style_imag1 = values['style_imag1']
    style_imag1 = download_file(url=style_imag1, save_dir='/content/ComfyUI/input', file_name='style_imag1')
    style_imag2 = values['style_imag2']
    style_imag2 = download_file(url=style_imag2, save_dir='/content/ComfyUI/input', file_name='style_imag2')
    style_imag3 = values['style_imag3']
    style_imag3 = download_file(url=style_imag3, save_dir='/content/ComfyUI/input', file_name='style_imag3')
    style_imag4 = values['style_imag4']
    style_imag4 = download_file(url=style_imag4, save_dir='/content/ComfyUI/input', file_name='style_imag4')
    positive_prompt = values['positive_prompt']
    negative_prompt = values['negative_prompt']
    seed = values['seed']
    steps = values['steps']
    cfg = values['cfg']
    sampler_name = values['sampler_name']
    scheduler = values['scheduler']
    width = values['width']
    height = values['height']
    enable_image_caption = values['enable_image_caption']
    if seed == 0:
        random.seed(int(time.time()))
        seed = random.randint(0, 18446744073709551615)
    input_image = LoadImage.load_image(input_image)[0]
    input_image = ImageScaleToMegapixels.image_scale_down_to_total_pixels(input_image, megapixels=1.0)[0]
    input_image, input_mask = BiRefNetUltraV2.birefnet_ultra_v2(input_image, birefnet_model, detail_method="VITMatte", detail_erode=4, detail_dilate=2, black_point=0.01, white_point=0.99, process_detail=False, device="cuda", max_megapixels=2.0)
    input_image = ImageRemoveAlpha.image_remove_alpha(input_image, fill_background=True, background_color="#FFFFFF", mask=input_mask)[0]
    input_image = ImageDesaturate.execute(input_image, factor=1.0, method="luminance (Rec.601)")[0]
    if enable_image_caption:
        caption_type = values['caption_type']
        caption_length = values['caption_length']
        low_vram = values['low_vram']
        positive_prompt = Joy_caption_two.generate(joy_two_pipeline, input_image, caption_type, caption_length, low_vram)[0]
    style_imag1 = LoadImage.load_image(style_imag1)[0]
    style_imag2 = LoadImage.load_image(style_imag2)[0]
    style_imag3 = LoadImage.load_image(style_imag3)[0]
    style_imag4 = LoadImage.load_image(style_imag4)[0]
    batch_image1 = ImageBatch.batch(style_imag1, style_imag2)[0]
    batch_image2 = ImageBatch.batch(style_imag3, style_imag4)[0]
    batch_image3 = ImageBatch.batch(batch_image1, batch_image2)[0]
    ip_unet = AV_IPAdapter.apply_ip_adapter("ip-adapter_sdxl_vit-h.safetensors", "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors", lora_unet, batch_image3, weight=1.5, weight_type="style transfer", start_at=0, end_at=1)[0]
    canny_image = AV_ControlNetPreprocessor.detect_controlnet(input_image, preprocessor="canny", sd_version="sdxl", resolution=640, preprocessor_override="None")[0]
    positive = CLIPTextEncode.encode(clip, positive_prompt)[0]
    negative = CLIPTextEncode.encode(clip, negative_prompt)[0]
    positive, negative = ControlNetApplyAdvanced.apply_controlnet(positive, negative, control_net, canny_image, strength=0.65, start_percent=0.0, end_percent=0.91, vae=vae)
    latent_image = EmptyLatentImage.generate(width, height, batch_size=1)[0]
    samples = KSampler.sample(ip_unet, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0)[0]
    decoded = VAEDecode.decode(vae, samples)[0].detach()
    Image.fromarray(np.array(decoded*255, dtype=np.uint8)[0]).save(f"/content/sdxl-line-art-style-transfer-{seed}-tost.png")

    result = f"/content/sdxl-line-art-style-transfer-{seed}-tost.png"
    try:
        notify_uri = values['notify_uri']
        del values['notify_uri']
        notify_token = values['notify_token']
        del values['notify_token']
        discord_id = values['discord_id']
        del values['discord_id']
        if(discord_id == "discord_id"):
            discord_id = os.getenv('com_camenduru_discord_id')
        discord_channel = values['discord_channel']
        del values['discord_channel']
        if(discord_channel == "discord_channel"):
            discord_channel = os.getenv('com_camenduru_discord_channel')
        discord_token = values['discord_token']
        del values['discord_token']
        if(discord_token == "discord_token"):
            discord_token = os.getenv('com_camenduru_discord_token')
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result)
        with open(result, "rb") as file:
            files = {default_filename: file.read()}
        payload = {"content": f"{json.dumps(values)} <@{discord_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{discord_channel}/messages",
            data=payload,
            headers={"Authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
        result_url = response.json()['attachments'][0]['url']
        notify_payload = {"jobId": job_id, "result": result_url, "status": "DONE"}
        web_notify_uri = os.getenv('com_camenduru_web_notify_uri')
        web_notify_token = os.getenv('com_camenduru_web_notify_token')
        if(notify_uri == "notify_uri"):
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
        else:
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        return {"jobId": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            if(notify_uri == "notify_uri"):
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            else:
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
                requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists(result):
            os.remove(result)
        if os.path.exists(input_image1):
            os.remove(input_image1)
        if os.path.exists(input_image2):
            os.remove(input_image2)

runpod.serverless.start({"handler": generate})