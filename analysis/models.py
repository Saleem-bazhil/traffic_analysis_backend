from django.db import models


class Analysis(models.Model):
    """Model for traffic analysis results."""

    FILE_TYPE_CHOICES = [
        ('image', 'Image'),
        ('video', 'Video'),
    ]

    filename = models.CharField(max_length=255)
    file_type = models.CharField(max_length=10, choices=FILE_TYPE_CHOICES)
    upload_path = models.CharField(max_length=255)
    result_path = models.CharField(max_length=255)
    vehicle_count = models.IntegerField()
    density_percentage = models.FloatField()
    processed_at = models.DateTimeField(auto_now_add=True)
    time_series_data = models.JSONField(null=True, blank=True)

    # Vehicle type counts
    car_count = models.IntegerField(default=0)
    truck_count = models.IntegerField(default=0)
    bus_count = models.IntegerField(default=0)
    motorcycle_count = models.IntegerField(default=0)

    # Direction-specific statistics
    north_count = models.IntegerField(default=0)
    south_count = models.IntegerField(default=0)
    east_count = models.IntegerField(default=0)
    west_count = models.IntegerField(default=0)

    north_density = models.FloatField(default=0.0)
    south_density = models.FloatField(default=0.0)
    east_density = models.FloatField(default=0.0)
    west_density = models.FloatField(default=0.0)

    class Meta:
        db_table = 'analysis'
        ordering = ['-processed_at']
        verbose_name_plural = 'Analyses'

    def __repr__(self):
        return f"<Analysis {self.id} - {self.filename}>"

    def __str__(self):
        return f"Analysis #{self.id}: {self.filename}"
