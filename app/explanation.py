import os
import base64
from mistralai import Mistral

def encode_image_base64(image_path):
    """Encodes an image to a base64 string for Mistral."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def generate_explanation(video_path: str, predicted_class: str, heatmap_path: str = None) -> str:
    """
    Calls Mistral (Pixtral Vision) to explain the physical action predicted.
    Note: Mistral's vision model takes images rather than full video files, 
          so we send the keyframe (heatmap) as the visual context.
    """
    # PASTE YOUR MISTRAL API KEY HERE
    api_key = "8cOgWI0Lon37PP8d6sd1SjWXi7T5xdbp"
    
    client = Mistral(api_key=api_key)
    
    prompt = f"A computer vision model just predicted the action '{predicted_class}' in a live video feed."
    
    messages = []
    
    if heatmap_path and os.path.exists(heatmap_path):
        prompt += " I have provided a heatmap (keyframe) showing the human subject and where the model was looking."
        prompt += " Briefly explain why this prediction makes sense based on the visual evidence. Keep it to 2-3 sentences max."
        
        base64_image = encode_image_base64(heatmap_path)
        if base64_image:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64_image}" 
                        }
                    ]
                }
            ]
    
    # Fallback to pure text if image processing fails
    if not messages:
        messages = [{"role": "user", "content": prompt + " Briefly describe what biological motions typically define this action in 2 sentences."}]
    
    try:
        chat_response = client.chat.complete(
            model="pixtral-12b-2409",
            messages=messages,
        )
        return chat_response.choices[0].message.content.replace('\n', ' ').strip()
        
    except Exception as e:
        return f"Explanation generation failed: {str(e)}"
