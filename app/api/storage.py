import aiofiles, os, uuid

async def save_upload(upload_file, upload_dir: str):
    ext = os.path.splitext(upload_file.filename)[1]
    filename = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(upload_dir, filename)

    async with aiofiles.open(path, "wb") as out:
        while True:
            chunk = await upload_file.read(1024*1024)
            if not chunk:
                break
            await out.write(chunk)
    return path