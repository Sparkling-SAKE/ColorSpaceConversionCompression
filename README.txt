# 환경 세팅
    필요한 라이브러리를 설치합니다. (neuralcompression, wandb, ...)
    - pip install -r requirements.txt

    dataset.py
    - (저는 openimage 를 사용해서 수정이 필요할 수 있습니다.)
    - Vimeo-90k (82GB) 데이터셋을 사용하기 위한 클래스를 제공합니다.
    - 다운로드 링크: http://toflow.csail.mit.edu/


# 학습 진행 
    gained_cheng.py
    - 구현한 모델을 Pytorch-Lightning 프레임워크를 이용하여 학습합니다. (train.py)

    lightning_model.py
    - LightningModel 클래스는 CompressAI의 모델을 Pytorch-Lightning 프레임워크로 래핑하는 클래스입니다.

    train.py
    - 모델을 학습할 때 사용할 파일이며 config/base.yaml 파일을 입력으로 하여 학습 조건 등을 설정할 수 있습니다.
    - 실행 커맨드 예시 (주의: config/base.yaml 파일 내에 다운받은 Vimeo-90k 데이터셋 디렉토리를 설정해준 후 사용 e.g. data_dir: "/home/gpuadmin/data/vimeo_septuplet" )
        CUDA_VISIBLE_DEVICES=1 python train.py
    - logging을 위해 Pytorch-Lightning에서 지원하는 wandb를 사용합니다. 위의 학습 커맨드 입력하면 나오는 안내를 따라주시기 바랍니다.

# 테스트 진행
    test.py
    - 학습이 끝나면, 최종적으로 가장 좋은 성능을 기록한 체크포인트가 "last.ckpt"에 저장됩니다.
    - test.py 파일을 통해 kodak24 데이터셋에 대하여 Rate-Distortion 성능을 평가합니다.
    - 실행 커맨드 예시 (Optinal: --write_recon_image 1)
        CUDA_VISIBLE_DEVICES=1 python test.py --checkpoint ./last.ckpt 

# 실습 발표
    - 출력된 PSNR와 BPP 결과를 CompressAI에서 제공하는 결과와 비교합니다.
            	gaind_cheng
        |    BPP            PSNR    |
        |0.123921712	27.699473   |
        |0.197981093	29.35881842 |
        |0.307244195	31.12891625 |
        |0.460611979	32.95067519 |
        |0.661665175	34.96829655 |
        |0.914404975	36.91109492 |
        |1.234208849	38.83751427 |
        |1.619089762	40.63331267 |

    - 추가 옵션 [--write_recon_image 1]를 이용하여 복원된 이미지를 확인합니다.
    - 실습을 진행하면서 생긴 질문사항 등을 정리합니다.
