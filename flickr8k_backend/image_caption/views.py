from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .utils.preprocess import preprocess_image_file
from .utils.caption_generator import generate_caption

@csrf_exempt
def predict_caption(request):
    """
    Handle image upload and return caption (plain Django version — guaranteed to work).
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method (use POST)'}, status=405)

    # Check for uploaded file
    image_file = request.FILES.get('file')
    if not image_file:
        return JsonResponse({'error': "No file uploaded (use key 'file')"}, status=400)

    try:
        # Preprocess image
        img_tensor = preprocess_image_file(image_file)

        # Generate caption
        caption = generate_caption(img_tensor)

        return JsonResponse({'caption': caption})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
