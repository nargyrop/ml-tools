from dl_tools.superres_models import RedNetModel

rednet = RedNetModel((128, 128, 1), filters=8).build_model()
print(rednet.summary())