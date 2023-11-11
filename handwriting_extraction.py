
from google.cloud import vision
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]='inkterpretor.json' 
#(created in step 1)


def detect_document(path):
    """Detects document features in an image."""

    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)

    if response.error.message:
        raise Exception(
        "{}\nFor more info on error messages, check: "
        "https://cloud.google.com/apis/design/errors".format(response.error.message)
    )

    text = response.text_annotations[0].description
    return text

if __name__ == "__main__":
    print(detect_document("detect_handwriting_OCR-detect-handwriting_SMALL.png"))
