from django.core.files.storage import default_storage
from django.shortcuts import render

from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.response import Response
from rest_framework.views import APIView

import io

from ultralytics import YOLO
from pathlib import Path

from django.conf import settings


# Create your views here.
class PillDetectorAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser, JSONParser,)
    serializer_class = None
    model_class = None

    def get_queryset(self):
        pass

    def post(self, request, *args, **kwargs):
        data = request.data
        print(data["pill_image"])
        image_path = default_storage.save(
            f'temporary_storage/{data["pill_image"].name}', 
            data['pill_image']
        )
        
        full_image_path = str(Path(settings.MEDIA_ROOT) / image_path)
        model = YOLO('yolov8n.pt')
        
        #results = model(settings.BASE_DIR / settings.MEDIA_ROOT / image_path)
        results = model(full_image_path)

        print(results[0].boxes.cls)

        default_storage.delete(image_path)

        return Response(status=status.HTTP_200_OK)
    

def pill_detector_test(request):
    return render(request, 'pill/container/pill_detector_test.html')