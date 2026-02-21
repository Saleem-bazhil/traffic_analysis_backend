from rest_framework import serializers
from django.conf import settings
from .models import Analysis


class AnalysisSerializer(serializers.ModelSerializer):
    """Full serializer for analysis list/detail views."""

    vehicle_counts = serializers.SerializerMethodField()
    lane_counts = serializers.SerializerMethodField()
    lane_densities = serializers.SerializerMethodField()
    upload_url = serializers.SerializerMethodField()
    result_url = serializers.SerializerMethodField()

    class Meta:
        model = Analysis
        fields = [
            'id', 'filename', 'file_type', 'upload_path', 'result_path',
            'vehicle_count', 'density_percentage', 'processed_at',
            'car_count', 'truck_count', 'bus_count', 'motorcycle_count',
            'north_count', 'south_count', 'east_count', 'west_count',
            'north_density', 'south_density', 'east_density', 'west_density',
            'time_series_data',
            'vehicle_counts', 'lane_counts', 'lane_densities',
            'upload_url', 'result_url',
        ]
        read_only_fields = fields

    def get_vehicle_counts(self, obj):
        return {
            'car': obj.car_count or 0,
            'truck': obj.truck_count or 0,
            'bus': obj.bus_count or 0,
            'motorcycle': obj.motorcycle_count or 0,
        }

    def get_lane_counts(self, obj):
        return {
            'North': obj.north_count or 0,
            'South': obj.south_count or 0,
            'East': obj.east_count or 0,
            'West': obj.west_count or 0,
        }

    def get_lane_densities(self, obj):
        return {
            'North': float(obj.north_density or 0.0),
            'South': float(obj.south_density or 0.0),
            'East': float(obj.east_density or 0.0),
            'West': float(obj.west_density or 0.0),
        }

    def get_upload_url(self, obj):
        request = self.context.get('request')
        if request and obj.upload_path:
            return request.build_absolute_uri(f'{settings.MEDIA_URL}uploads/{obj.upload_path}')
        return None

    def get_result_url(self, obj):
        request = self.context.get('request')
        if request and obj.result_path:
            return request.build_absolute_uri(f'{settings.MEDIA_URL}results/{obj.result_path}')
        return None


class AnalysisUploadSerializer(serializers.Serializer):
    """Serializer for file upload validation."""

    file = serializers.FileField()
    conf_global = serializers.FloatField(default=0.6, required=False)
    motorcycle_conf = serializers.FloatField(default=0.75, required=False)
    iou_thresh = serializers.FloatField(default=0.3, required=False)

    def validate_file(self, value):
        # Validate extension
        ext = value.name.rsplit('.', 1)[-1].lower() if '.' in value.name else ''
        if ext not in settings.ALLOWED_UPLOAD_EXTENSIONS:
            raise serializers.ValidationError(
                f'Invalid file type. Allowed: {", ".join(settings.ALLOWED_UPLOAD_EXTENSIONS)}'
            )

        # Validate file size
        if value.size > settings.MAX_FILE_SIZE:
            max_mb = settings.MAX_FILE_SIZE // (1024 * 1024)
            raise serializers.ValidationError(
                f'File too large. Maximum size is {max_mb}MB.'
            )

        return value


class SignalConfigSerializer(serializers.Serializer):
    """Serializer for signal timing configuration."""

    green_min = serializers.IntegerField(min_value=1, max_value=60)
    green_max = serializers.IntegerField(min_value=1, max_value=120)
    yellow = serializers.IntegerField(min_value=1, max_value=10)
    red_min = serializers.IntegerField(min_value=1, max_value=30)
    red_max = serializers.IntegerField(min_value=1, max_value=60)
    min_green = serializers.IntegerField(min_value=0, max_value=60)
    hysteresis = serializers.IntegerField(min_value=0, max_value=30)

    def validate(self, data):
        if data.get('green_min', 0) >= data.get('green_max', 999):
            raise serializers.ValidationError('green_min must be less than green_max')
        if data.get('red_min', 0) >= data.get('red_max', 999):
            raise serializers.ValidationError('red_min must be less than red_max')
        if data.get('min_green', 0) > data.get('green_max', 999):
            raise serializers.ValidationError('min_green must not exceed green_max')
        return data
