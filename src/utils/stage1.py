import numpy as np
from .inference import inference,get_binary_mask,mask_to_bbox
# 0-> background
# 1->pagenum
# 2->QN
# 3->signature
# 4->student info
def s1_binary_masks(s1_mask):
    page_mask = get_binary_mask(s1_mask,1)
    qn_mask = get_binary_mask(s1_mask,2)
    sig_mask = get_binary_mask(s1_mask,3)
    info_mask = get_binary_mask(s1_mask,4)
    return np.array([page_mask,qn_mask, sig_mask, info_mask])
def get_question_numbers(image,bbox):    
    questions = []
    for b in bbox:
        x1,y1,w,h=b
        roi=np.array(image[y1:y1+h,x1:x1+w])
        questions.append(roi)
    return np.array(questions)




    



