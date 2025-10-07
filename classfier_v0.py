from IPython.core.display import Image
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
import torchvision
import torch

weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model = model.eval().cuda()


preprocess = weights.transforms()
#input image w x h x c

def resnet_predict(image):
    with torch.no_grad():
      prediction = resnet_predict_raw(image)
      if prediction.shape[0] == 1:
        prediction = prediction.squeeze(0).softmax(0)
        class_id = prediction.argmax().item()
        score = prediction[class_id].item()
        category_name = weights.meta["categories"][class_id]
        # return(f"class id - {class_id} {category_name}: {100 * score:.1f}%")
        return(f"{category_name}: {100 * score:.3f}%")
      res_lst = []
      for p in prediction:
        p = p.squeeze(0).softmax(0)
        class_id = p.argmax().item()
        score = p[class_id].item()
        category_name = weights.meta["categories"][class_id]
        res_lst.append(category_name)
      return res_lst


def resnet_predict_raw(image):


  if image.shape != (3, 256, 256):
    rimage = torchvision.transforms.Resize((256, 256))(image)
  else:
    rimage = image

  # Step 3: Apply inference preprocessing transforms
  batch = preprocess(rimage)

  # Step 4: Use the model and print the predicted category
  return model(batch).softmax(-1)


batch_size = 1
orig_clases = torch.tensor([817, 705, 609, 586, 436, 627, 468, 621, 803, 407, 408, 751, 717,866, 661]).cuda()
total_clases_without_orig = torch.tensor([x for x in list(range(0, 1000)) if x not in orig_clases]).cuda()


# def adv_loss_calc(image):
#     adv_loss = 0
#     pred = resnet_predict_raw(image)
#     for p in pred:
#       adv_loss += torch.stack([100*p.softmax(0)[c.item()] for c in orig_clases]).mean() / pred.shape[0]
#     return adv_loss

def adv_loss_calc(image):
    assert len(image.shape) == 4, "Image should be of shape (batch_size, 3, h, w)"
    adv_loss = []
    pred = resnet_predict_raw(image)
    for p in pred:
      adv_loss.append(p[orig_clases].mean())
    return torch.stack(adv_loss)



def adv_loss_calc2(image):
    adv_loss = []
    pred = resnet_predict_raw(image)
    for p in pred:
      forbiden = p[orig_clases].max()
      allowed = p[total_clases_without_orig].max()
      adv_loss.append(forbiden - allowed)
    return torch.stack(adv_loss)