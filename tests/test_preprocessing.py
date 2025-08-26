import io
import zipfile
from PIL import Image
from backend.preprocessing import handle_uploaded_file

class DummyUpload:
    """Mock Streamlit uploaded file object."""
    def __init__(self, name, content):
        self.name = name
        self._content = content
    def getbuffer(self):
        return self._content

def test_single_image():
    # Create a fake image
    img = Image.new("RGB", (100, 100), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    uploaded = DummyUpload("test.png", buf.getvalue())
    patches = handle_uploaded_file(uploaded)

    assert len(patches) == 1
    assert isinstance(patches[0], Image.Image)

def test_zip_images():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        # Add one fake image
        img = Image.new("RGB", (50, 50), color="blue")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        z.writestr("patch1.jpg", img_bytes.getvalue())

    buf.seek(0)
    uploaded = DummyUpload("patches.zip", buf.getvalue())
    patches = handle_uploaded_file(uploaded)

    assert len(patches) == 1
    assert isinstance(patches[0], Image.Image)