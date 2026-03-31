from huggingface_hub import upload_file

def main():
    upload_file(
        path_or_fileobj="model.pth",
        path_in_repo="model.pth",
        repo_id="ItsLiang/Cat-Skin-Disease-Prediction",
        repo_type="model"
    )