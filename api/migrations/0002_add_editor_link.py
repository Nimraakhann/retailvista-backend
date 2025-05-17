from django.db import migrations, models

class Migration(migrations.Migration):

    dependencies = [
        ('api', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='map',
            name='editor_link',
            field=models.URLField(blank=True, null=True),
        ),
    ]