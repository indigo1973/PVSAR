import os
import shutup

shutup.please()


NITERS      = 30
target_config_path = "configs/lightbox.json"

for i in range(1, NITERS):

    # print("Training Loop Sunlamp - NITER ", i)
    print("Training Loop Lightbox - NITER ", i)

    os.system("python generate_labels.py --cfg " +  target_config_path)
    os.system("python loop_pesudo.py --cfg " +  target_config_path)
