from rest_framework import serializers
from .models import Promotion

class PromotionSerializer(serializers.ModelSerializer):
    image_url = serializers.SerializerMethodField()

    class Meta:
        model = Promotion
        fields = ['id', 'name', 'image', 'status', 'created_at', 'image_url']
        read_only_fields = ['id', 'created_at']

    def get_image_url(self, obj):
        if obj.image:
            return self.context['request'].build_absolute_uri(obj.image.url)
        return None 