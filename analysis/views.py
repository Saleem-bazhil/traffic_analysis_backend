import os
import csv
from django.conf import settings
from django.http import HttpResponse, FileResponse
from rest_framework import viewsets, status, views, generics
from rest_framework.response import Response
from rest_framework.decorators import action
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

from .models import Analysis
from .serializers import AnalysisSerializer, AnalysisUploadSerializer, SignalConfigSerializer
from .utils import process_image, process_video, generate_unique_filename, get_upload_path, get_result_path
from .signal_controller import generate_signal_plan

class AnalysisViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows analyses to be viewed or deleted.
    """
    queryset = Analysis.objects.all().order_by('-processed_at')
    serializer_class = AnalysisSerializer

    @action(detail=False, methods=['delete'])
    def clear(self, request):
        """Clear all analysis history."""
        # Optional: delete files from disk as well
        for analysis in Analysis.objects.all():
            if analysis.upload_path:
                upload_file = os.path.join(get_upload_path(), analysis.upload_path)
                if os.path.exists(upload_file):
                    os.remove(upload_file)
            if analysis.result_path:
                result_file = os.path.join(get_result_path(), analysis.result_path)
                if os.path.exists(result_file):
                    os.remove(result_file)
        
        count, _ = Analysis.objects.all().delete()
        return Response({'message': f'Cleared {count} analyses.'}, status=status.HTTP_200_OK)

    @action(detail=True, methods=['get'])
    def signals(self, request, pk=None):
        """Get signal plan for a specific analysis."""
        analysis = self.get_object()
        lane_densities = {
            'North': analysis.north_density,
            'South': analysis.south_density,
            'East': analysis.east_density,
            'West': analysis.west_density,
        }
        plan = generate_signal_plan(lane_densities)
        return Response({'signal_plan': plan})

    @action(detail=True, methods=['get'])
    def densities(self, request, pk=None):
        """Get lane densities for a specific analysis."""
        analysis = self.get_object()
        densities = {
            'North': analysis.north_density,
            'South': analysis.south_density,
            'East': analysis.east_density,
            'West': analysis.west_density,
        }
        return Response({'lane_densities': densities})

    @action(detail=True, methods=['get'], url_path='report/csv')
    def report_csv(self, request, pk=None):
        """Download analysis report as CSV."""
        analysis = self.get_object()
        
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="analysis_{analysis.id}.csv"'
        
        writer = csv.writer(response)
        writer.writerow(['Parameter', 'Value'])
        writer.writerow(['ID', analysis.id])
        writer.writerow(['Filename', analysis.filename])
        writer.writerow(['Date', analysis.processed_at.strftime("%Y-%m-%d %H:%M:%S")])
        writer.writerow(['Total Vehicles', analysis.vehicle_count])
        writer.writerow(['Overall Density (%)', f"{analysis.density_percentage:.2f}"])
        writer.writerow(['Cars', analysis.car_count])
        writer.writerow(['Trucks', analysis.truck_count])
        writer.writerow(['Buses', analysis.bus_count])
        writer.writerow(['Motorcycles', analysis.motorcycle_count])
        writer.writerow(['North Density (%)', f"{analysis.north_density:.2f}"])
        writer.writerow(['South Density (%)', f"{analysis.south_density:.2f}"])
        writer.writerow(['East Density (%)', f"{analysis.east_density:.2f}"])
        writer.writerow(['West Density (%)', f"{analysis.west_density:.2f}"])
        
        return response

    @action(detail=True, methods=['get'], url_path='report/json')
    def report_json(self, request, pk=None):
        """Download analysis report as JSON."""
        analysis = self.get_object()
        serializer = self.get_serializer(analysis)
        response = Response(serializer.data)
        response['Content-Disposition'] = f'attachment; filename="analysis_{analysis.id}.json"'
        return response

    @action(detail=True, methods=['get'])
    def snapshot(self, request, pk=None):
        """Download snapshot image."""
        analysis = self.get_object()
        if not analysis.result_path:
            return Response({'error': 'No result file found'}, status=status.HTTP_404_NOT_FOUND)
            
        result_file = os.path.join(get_result_path(), analysis.result_path)
        if not os.path.exists(result_file):
            return Response({'error': 'File not found on disk'}, status=status.HTTP_404_NOT_FOUND)
            
        return FileResponse(open(result_file, 'rb'), as_attachment=True, filename=f"snapshot_{analysis.filename}")

    @action(detail=False, methods=['get'])
    def sample(self, request):
        """Get sample analysis data for demo purposes."""
        sample_data = {
            'total_vehicles': 142,
            'density': 68.5,
            'vehicle_counts': {
                'car': 85,
                'truck': 24,
                'bus': 12,
                'motorcycle': 21
            },
            'lane_counts': {
                'North': 45,
                'South': 62,
                'East': 35,
                'West': 50
            },
            'lane_densities': {
                'North': 75.0,
                'South': 92.5,
                'East': 58.3,
                'West': 83.3
            }
        }
        return Response(sample_data)

class AnalysisUploadView(views.APIView):
    """
    API endpoint for uploading and processing files.
    """
    serializer_class = AnalysisUploadSerializer
    parser_classes = [rest_framework.parsers.MultiPartParser if hasattr(views, 'rest_framework') else __import__('rest_framework.parsers').parsers.MultiPartParser]

    def post(self, request, *args, **kwargs):
        serializer = AnalysisUploadSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            
        file_obj = serializer.validated_data['file']
        
        # Determine file type
        filename = file_obj.name
        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
        file_type = 'video' if ext in ['mp4', 'avi', 'mov', 'webm'] else 'image'
        
        # Save file to uploads directory
        unique_filename = generate_unique_filename(filename)
        upload_dir = get_upload_path()
        upload_filepath = os.path.join(upload_dir, unique_filename)
        
        with open(upload_filepath, 'wb+') as destination:
            for chunk in file_obj.chunks():
                destination.write(chunk)
                
        # Generate result filename
        result_filename = f"result_{unique_filename}"
        if file_type == 'video' and not result_filename.endswith('.webm'):
            result_filename = result_filename.rsplit('.', 1)[0] + '.webm'
            
        result_dir = get_result_path()
        result_filepath = os.path.join(result_dir, result_filename)
        
        try:
            # Process the file
            if file_type == 'image':
                stats = process_image(upload_filepath, result_filepath)
                # Map lane_counts to model fields (process_image returns totals)
                lane_stats = stats['lane_counts']
            else:
                conf_global = serializer.validated_data.get('conf_global', 0.6)
                motorcycle_conf = serializer.validated_data.get('motorcycle_conf', 0.75)
                iou_thresh = serializer.validated_data.get('iou_thresh', 0.3)
                
                stats = process_video(
                    upload_filepath, 
                    result_filepath,
                    conf_global=conf_global,
                    motorcycle_conf=motorcycle_conf,
                    iou_thresh=iou_thresh
                )
                lane_stats = stats.get('lane_avg_counts', stats.get('lane_counts', {}))
                
            # Create database record
            analysis = Analysis.objects.create(
                filename=filename,
                file_type=file_type,
                upload_path=unique_filename,
                result_path=result_filename,
                vehicle_count=stats['total_vehicles'],
                density_percentage=stats['density'],
                car_count=stats['vehicle_counts']['car'],
                truck_count=stats['vehicle_counts']['truck'],
                bus_count=stats['vehicle_counts']['bus'],
                motorcycle_count=stats['vehicle_counts']['motorcycle'],
                north_count=int(lane_stats.get('North', 0)),
                south_count=int(lane_stats.get('South', 0)),
                east_count=int(lane_stats.get('East', 0)),
                west_count=int(lane_stats.get('West', 0)),
                north_density=stats['lane_densities'].get('North', 0.0),
                south_density=stats['lane_densities'].get('South', 0.0),
                east_density=stats['lane_densities'].get('East', 0.0),
                west_density=stats['lane_densities'].get('West', 0.0),
                time_series_data=stats.get('time_series_data', [])
            )
            
            # Return serialized analysis
            response_serializer = AnalysisSerializer(analysis, context={'request': request})
            return Response(response_serializer.data, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            # Clean up files on error
            if os.path.exists(upload_filepath):
                os.remove(upload_filepath)
            if os.path.exists(result_filepath):
                os.remove(result_filepath)
                
            return Response(
                {'error': str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class SignalConfigView(views.APIView):
    """
    API endpoint to get and update signal configuration.
    """
    def get(self, request):
        config = getattr(settings, 'DEFAULT_SIGNAL_CONFIG', {})
        # Note: In a real app, this might come from a database model
        # For now, we return the settings default
        return Response(config)
        
    def put(self, request):
        serializer = SignalConfigSerializer(data=request.data)
        if serializer.is_valid():
            # In a real app, save to database
            # For this port, we just echo back the valid config 
            return Response(serializer.validated_data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
