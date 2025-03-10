import concurrent.futures
import json
from typing import List
import boto3
from boto3.s3.transfer import TransferConfig
import os
import concurrent
from dotenv import load_dotenv
from pathlib import Path

class S3Helper:
    def __init__(self, root_path: str, bucket_name: str, region_name: str='ap-northeast-2', local_root_path:str='model'):
        self.root_path=root_path
        self.bucket_name=bucket_name
        self.region_name=region_name
        self.save_s3_file_root_path=local_root_path

        with open(root_path + '/env/s3-bucket-key.json', 'r', encoding='utf-8-sig') as f:
            keys = json.load(f)

        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=keys['AccessKey'],
            aws_secret_access_key=keys['SecretAccessKey'],
            region_name=region_name
        )

        chunk_size = 100 * 1024 * 1024
        self.multipart_config = TransferConfig(multipart_threshold=chunk_size,
                                               multipart_chunksize=chunk_size,
                                               max_concurrency=10,
                                               use_threads=True)

    def upload(self, local_fpaths: List[str] = [], s3_fpaths: List[str] = [], from_local: bool=False):
        if from_local:
            local_fpaths = [str(file) for file in Path('model').rglob('*') if file.is_file() and file.name != 'dataset.csv']
            s3_fpaths = [fpath[len(self.save_s3_file_root_path)+1:] for fpath in local_fpaths]
            
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._upload_file, local_fpath, s3_fpath) 
                for (local_fpath, s3_fpath) in zip(local_fpaths, s3_fpaths)
            ]

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"업로드 중 오류 발생: {e}")

    def _upload_file(self, local_fpath: str, s3_fpath: str):
        self.s3_client.upload_file(local_fpath, self.bucket_name, s3_fpath, Config=self.multipart_config)

    def download(self, s3_fpaths: List[str] = []):
        if (len(s3_fpaths) == 0):
            s3_fpaths = self._get_file_metadata()
            local_fpaths = [f'{self.save_s3_file_root_path}/{s3_fpath}' for s3_fpath in s3_fpaths]

        dir_paths = [f'{self.save_s3_file_root_path}/{fpath}' for fpath in set([str(Path(fpath).parent) for fpath in s3_fpaths])]
        for dir_path in dir_paths:
            path = Path(dir_path)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._download_file, s3_fpath, local_fpath) 
                for (local_fpath, s3_fpath) in zip(local_fpaths, s3_fpaths)
            ]

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"다운로드 중 오류 발생: {e}")

    def _download_file(self, s3_fpath: str, local_fpath: str):
        self.s3_client.download_file(self.bucket_name,
                                     s3_fpath,
                                     local_fpath,
                                     Config=self.multipart_config)

    def print_list(self, root_folder_name=''):
        content_names = self._get_file_metadata(root_folder_name)
        for cotent_name in content_names:
            print(cotent_name)

    def _get_file_metadata(self, root_folder_name=''):
        response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=root_folder_name)
        return [file['Key'] for file in response.get('Contents', [])]






if __name__ == "__main__":
    load_dotenv(os.path.join(os.path.join(os.path.dirname(__file__), "env", ".env")))

    root_path = os.path.dirname(__file__)

    s3_helper = S3Helper(root_path, 'youtube-comment-predict', 'ap-northeast-2')

    s3_helper.print_list()

    # s3_helper.upload(from_local=True)
    s3_helper.download()