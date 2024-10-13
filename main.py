import argparse
import math
import boto3
from io import BytesIO
from datetime import datetime
from PIL import Image


class AWSImageRecognition:
    RATIO = (3, 2)

    def __init__(self, temp_path, image_name, aws_access_key_id, aws_secret_access_key, aws_session_token, region_name, bucket):
        self.temp_path = temp_path
        self.image_name = image_name

        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region_name=region_name,
        )
        self.rekognition_client = session.client('rekognition')
        self.s3_client = session.client('s3')
        self.bucket = bucket

    def process(self):
        top_left, bottom_right = self.detect_labels_and_coordinates()

        image_object = self.s3_client.get_object(Bucket=self.bucket, Key=self.image_name)
        image_data = image_object['Body'].read()

        with Image.open(BytesIO(image_data)) as img:
            area_width, area_height = self.get_image_dimensions(img, top_left, bottom_right)
            shift_horizontal, shift_vertical = self._calculate_shifts(area_width, area_height)
            cropped_image = self.crop_image(img, top_left, bottom_right, shift_horizontal, shift_vertical)
            self.save_image(cropped_image)

    def _calculate_shifts(self, width, height):
        shift_horizontal = shift_vertical = 0
        if width < height:
            target_width = math.floor(height / self.RATIO[0] * self.RATIO[1])
            shift_horizontal = (target_width - width) / 2
        elif width > height:
            target_height = math.floor(width / self.RATIO[0] * self.RATIO[1])
            shift_vertical = (target_height - height) / 2

        return shift_horizontal, shift_vertical

    def get_image_dimensions(self, img, top_left, bottom_right):
        width, height = img.size

        left = int(top_left[0] * width)
        top = int(top_left[1] * height)
        right = int(bottom_right[0] * width)
        bottom = int(bottom_right[1] * height)

        area_width = right - left
        area_height = bottom - top

        return area_width, area_height

    def detect_labels_and_coordinates(self):
        response = self.rekognition_client.detect_labels(Image={'S3Object': {'Bucket': self.bucket, 'Name': self.image_name}}, MaxLabels=10)

        top_left = (float('inf'), float('inf'))
        bottom_right = (float('-inf'), float('-inf'))

        for label in response['Labels']:
            for instance in label['Instances']:
                box = instance['BoundingBox']
                left = box['Left']
                top = box['Top']
                right = left + box['Width']
                bottom = top + box['Height']

                top_left = (min(top_left[0], left), min(top_left[1], top))
                bottom_right = (max(bottom_right[0], right), max(bottom_right[1], bottom))

        return top_left, bottom_right

    def crop_image(self, img, top_left, bottom_right, shift_horizontal=0, shift_vertical=0):
        width, height = img.size

        left = int(top_left[0] * width)
        top = int(top_left[1] * height)
        right = int(bottom_right[0] * width)
        bottom = int(bottom_right[1] * height)

        new_width = right - left + shift_horizontal * 2
        new_height = bottom - top + shift_vertical * 2

        left = max(0, left - shift_horizontal)
        right = min(width, left + new_width)
        top = max(0, top - shift_vertical)
        bottom = min(height, top + new_height)

        cropped_img = img.crop((left, top, right, bottom))

        return cropped_img

    def save_image(self, image, prefix=''):
        current_time = datetime.now().strftime("%d%m%H%M%S")
        temp_file_path = f"{self.temp_path}/{prefix}{current_time}_{self.image_name}"
        image.save(temp_file_path, "JPEG")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--temp_path', required=True, help='Path for temporary image files')
    parser.add_argument('--image_name', required=True, help='Name of the image in S3')
    parser.add_argument('--aws_access_key_id', required=True, help='AWS access key ID')
    parser.add_argument('--aws_secret_access_key', required=True, help='AWS secret access key')
    parser.add_argument('--aws_session_token', required=True, help='AWS session token')
    parser.add_argument('--region_name', required=True, help='AWS region name')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')

    args = parser.parse_args()

    aws = AWSImageRecognition(
        temp_path=args.temp_path,
        image_name=args.image_name,
        aws_access_key_id=args.aws_access_key_id,
        aws_secret_access_key=args.aws_secret_access_key,
        aws_session_token=args.aws_session_token,
        region_name=args.region_name,
        bucket=args.bucket
    )
    aws.process()


if __name__ == '__main__':
    main()
