import json
from random import shuffle
import os
from PIL import Image

# # data = json.load(open('/root/VQA_Main/Modified_MedVQA-main/data/VQA_RAD Dataset Public.json', 'r'))
# test = json.load(open('/root/VQA_Main/Modified_MedVQA-main/data/testset.json', 'r'))
# test_answer = [str(i["answer"]).lower for i in test]

# train = json.load(open('/root/VQA_Main/Modified_MedVQA-main/data/trainset.json', 'r'))
# train_answer = [str(i["answer"]).lower for i in train]

# data = test + train
# all_answers = list(set(i["answer"] for i in data))
# [str(i).lower for i in all_answers]
# print("=============answer len of all ",len(all_answers))

# closed = [i["answer"] for i in data if 'CLOSED' in i['answer_type']]
# closed_answers = list(set(closed))
# [str(i).lower for i in closed_answers]
# print("Closed question type number:", len(closed))

# opened_answer = list(set([i["answer"] for i in data if i['answer_type'] == 'OPEN']))
# [str(i).lower for i in opened_answer]
# print("=============answer len of open",len(opened_answer))


# # shuffle(closed)
# # shuffle(opened)

# # close_num = int(len(closed) * 0.87)
# # opened_num = int(len(opened) * 0.87)

# # train = closed[:close_num] + opened[:opened_num]
# # test = closed[close_num:] + opened[opened_num:]
# # train = json.load(open('/root/VQA_Main/Modified_MedVQA-main/data/trainset.json', 'r'))
# # test = json.load(open('/root/VQA_Main/Modified_MedVQA-main/data/testset.json', 'r'))

# # test_answer = [i["answer"] for i in test]

# annotation = {'train': train, 'test': test}
# json.dump(annotation, open('/root/VQA_Main/Modified_MedVQA-main/data/annotation.json', 'w'))

# #Modified
# # train_answer = []
# # for item in annotation["train"]:
# #     train_answer.append(item["answer"])
# # train_answer = list(set(train_answer))

# # test_answer = []
# # for item in annotation["test"]:
# #     test_answer.append(item["answer"])
# # test_answer = list(set(test_answer))
# # print('==========train answers length:', len(train_answer))
# # print('==========test answers length:', len(test_answer))

# # answer_query_list = list(answers_set)
# # print(len(answer_query_list))
# answer_query = {'answers': all_answers}



# json.dump(answer_query, open('/root/VQA_Main/Modified_MedVQA-main/data/answer_query.json', 'w'))


# # data = json.load(open('/root/VQA_Main/Modified_MedVQA-main/data/annotation.json', 'r'))
# # data = data['train'] + data['test']
# # answers = list(set([i['answer']for i in data]))
# ans2idx = {ans:idx for idx, ans in enumerate(all_answers)}
# json.dump(ans2idx, open('/root/VQA_Main/Modified_MedVQA-main/data/ans2idx.json', 'w'))
# ans2idx_closed = {ans:idx for idx, ans in enumerate(closed_answers)}
# print("Number of closed questions ", len(ans2idx_closed))
# # print("ans2idx_closed:", ans2idx_closed)
# json.dump(ans2idx_closed, open('/root/VQA_Main/Modified_MedVQA-main/data/ans2idx_closed.json', 'w'))


# ans2idx_opened = {ans:idx for idx, ans in enumerate(opened_answer)}
# print("Number of opened questions ",len(opened_answer))
# json.dump(ans2idx_opened, open('/root/VQA_Main/Modified_MedVQA-main/data/ans2idx_opened.json', 'w'))

# ans2label_dic = {'crescent': 397, '5.6cm focal predominantly hypodense': 349, 'heart lungs': 68, 'hydrocephalus': 199, 'right temporal lobe': 38, 'hypodense lesion': 14, 'brain': 8, 'stomach bubble': 211, 'abcess': 387, 'aorta and inferior vena cava': 161, 'suprasellar cistern': 53, 'left rectus abdominus': 247, 'left thalamus and basal ganglia': 189, 'no': 1, 'large bowel': 119, 'edematous': 84, 'underneath right hemidiaphragm': 326, 'both': 71, 'air fluid level': 209, 'female': 116, 'enlarged': 30, 'in vasculature': 287, 'rounded well defined pulmonary nodules varying in size and pattern': 429, 'width of aorta': 79, 'descending colon': 267, '6.5 x 6.2 x 8.8cm': 280, 'reduced sulci': 296, 'multiple sclerosis': 362, 'pacemaker': 91, 'cns': 131, 'exterior': 440, 'calcifications': 98, 'on left': 276, "superficial to patient's skin": 456, 'distal basilar artery': 250, 'epidural hematoma': 381, 'right posteroinferior cerebellum': 200, 'gallbladder': 203, '4th and 5th': 56, 'right side of trachea': 226, 'ring enhancing': 225, 'parietal and occipital lobes': 128, 'ring enhancing lesion in right frontal lobe': 146, 'basilar artery thrombosis': 312, 'fat accumulations': 112, 'outside': 441, 'normal': 330, 'left side': 64, 'posterior brain': 227, 'hydropneumothorax': 310, 'hyperintensity of left basal ganglia': 367, 'costophrenic angle blunting': 392, 'maxillary sinuses': 218, 'extraluminal air and small fluid collection': 33, 'high on image': 62, 'ivc': 166, 'mri flair': 252, 'extra axial and at right choroidal fissure': 35, 'chronic sinusitis vs hemorrhage': 302, 'exophytic cyst': 447, 'ring enhancing lesion in left occipital lobe': 127, '2.5cm x 1.7cm x 1.6cm': 317, 'left mca': 358, 'lateral ventricles': 114, 'right sided pleural effusion': 346, 'partial silhouetting': 368, 'surrounding tissue': 74, 'right lower lobe': 77, 'more dense': 72, 'lungs bony thoracic cavit y mediastinum and great vessels': 120, 'bilateral pleural effusion': 430, 'right colon': 241, 'ureteral obstruction': 377, 'viral inflammatory': 403, 'ultrasound': 2, 'breasts': 293, 'sacroiliac joint': 219, 'volume loss': 31, 'caudate putamen left parietal': 254, 'pneumonia': 366, 'pulmonary nodules': 12, 'abdomen': 152, 'lower lung fields': 333, 'bronchiectasis': 388, 'mri': 235, 'ruq pain jaundiceweight loss': 413, 'contrast': 102, '5 cm': 60, 'retroperitoneum retroperitoneal space': 121, 'right cerebellopontine angle': 196, 'both sides': 439, 'emphysema': 295, 'left costophrenic angle is blunted': 156, 'omental caking': 153, 'loculated': 167, 'right side': 63, 'on right shoulder': 90, 'posterior to gastric antrum': 338, 'aorta enhancement': 391, 'imagine patient is laying down and you are looking from feet': 164, 'right sided aortic arch': 182, 'chest': 28, 'temporal lobe': 214, 'ring enhancing lesion': 16, 'concave': 398, 'c t ratio': 386, 'hyperintense': 243, 'cystic duct is more tortuous': 417, 'csf is white': 177, 'pa': 55, 'anterior cerebrum': 255, 'pa xray': 191, 'well circumscribed': 446, 'cva': 404, 'subarachnoid': 162, 'right frontal lobe': 104, 'above': 54, 'pons': 288, 'it is less than half width of thorax': 69, 'asymmetric': 11, 'r hemidiaphragm': 284, 'chest xray': 9, 'right pica': 212, 'cerebellum': 183, 'nephroblastomatosis': 409, 'hip bones': 272, 'nipple location': 248, 'temporal and lateral occipital lobes': 274, 'on top of patient': 455, 'lung': 100, 'gadolinium': 314, '4': 277, 'sternal wires': 181, 'frontal and occipital': 303, 'below 7th rib in right lung': 233, 'contrast in intestines': 97, 'left aca and mca': 253, 'right lobe': 89, 'right lobe of liver': 47, 'iv contrast': 244, 'respiratory cardia c musculoskeletal': 176, 'sulcal effacement': 213, 'pres': 129, 'parasitic': 384, 'l2': 215, 't2 weighted': 188, 'pancreatic body': 46, 'extremities': 3, 'pineal gland': 169, 'blind ending loop of bowel arising from cecum': 237, 'sella and suprasellar cistern': 187, 'nodular opacities': 34, 'calcified atherosclerosis': 339, 'peritoneum': 360, 'aorta': 160, 'decreased muscle bulk': 279, 'fatty infiltration': 263, 'adjacent to vertebrae': 269, 'isointense': 242, 'imaging artifacts': 101, 'left cerebellum': 202, 'bilateral frontal lobes and body of corpus callosum': 249, 'cystic lesions': 42, 'bilateral parietal lobes': 301, 'hypodense': 21, 'mixed intensity': 275, 'cystic': 299, 'posterior horn of left lateral ventricle': 108, 'fat stranding around appendix thickened appendiceal walls dilated appendix and appendicolith is seen as well': 145, 'free air': 13, 'sternotomy wires and surgical clips': 139, 'bilateral': 24, 'location of contrast': 410, 'right hemisphere': 48, 'abdomen and pelvis': 230, 'cirrhosis': 25, 'white versus grey matter brightness': 424, 'cardiovascular': 5, 'fluid in pleural space': 393, 'bilateral cerebellum': 179, 'kidneys': 66, 'cartilage is not well viewed by xrays': 206, 'basal ganglia cerebellum cerebral cortex': 304, 'stones cancer infection anatomic variants': 419, 'pineal region': 238, 'occipital lobe': 106, 'mr t2 weighted': 262, 'portal vein': 351, 'bowel contents light up on image': 165, '7th rib': 81, 'superior': 61, 'ms plaques': 359, 'all 3 vascular distributions': 361, 'right cerebellum': 174, 'ap': 321, 'not sure': 407, 'white matter plaques': 305, 'cardiac region': 170, 'motor weakness sensory deficits and left neglect': 402, 'csf is brightly lit': 390, 'multilobulated': 273, 'prior surgery': 36, 'sharp costophrenic angles': 434, 'right subclavian vein': 329, 'varicocele': 220, 'psoas muscles': 380, 'small bowel': 67, 'liver': 109, 'vascular': 137, 'lungs': 23, 'if heart diameter is greater than half diameter of thoracic cavity': 385, 'lateral and third ventricular hydrocephalus': 231, 'gi': 15, 'abdominal pain': 414, 'kidney cyst': 260, 'posterior to appendix': 80, '10 20 minutes': 117, 'xray': 93, '3.4 cm': 186, 'infection': 426, 'in thorasic aorta': 78, 'quadrantopia aphasia memory deficit etc': 401, 'horsehoe kidney': 436, 'right upper lobe': 20, 'infarct': 171, 'non contrast ct': 157, '2': 291, 'right parietal lobe': 27, 'biconvex': 379, 'semi upright position': 433, 'plain film xray': 307, 'abscess': 41, 'infarcts': 449, 'right lung hilum': 87, 'nucleus pulposus': 457, 'xray plain film': 150, 'embolus': 450, 'basal ganglia': 50, 'plicae circulares': 421, 'suprasellar': 251, '5%': 428, 'right parietal': 435, 'small subdural hematoma with cerebral edema': 355, 'tumors gallstones': 418, 'right mca': 399, 'dwi': 195, 'left kidney': 70, 'oculomotor nerve cn iii and trigeminal nerves cn v': 190, 'right lateral ventricle': 194, 'left parietal lobe': 180, 'choroid plexus': 205, 'posteroanterior': 103, 'right': 40, 'elliptical': 222, 'cecum': 132, 'blunting of costophrenic angle loss of right hemidiaphragm and right heart border': 442, 'mass': 133, 'atherosclerotic calcification': 26, 'shrunken and nodular': 448, 't5': 57, 'man': 298, 'bit': 107, 'psoas major muscle': 336, 'upper right lobe': 96, 'left lung': 83, 'left mid lung': 143, 'retrocardiac': 232, 'cortical ribbon of right occipital lobe with extension into right posterior temporal lobe': 451, 'anterior surface': 256, 'white matter': 118, 'left hemisphere': 88, 'flair': 324, 'l2 3': 216, '3rd rib': 149, 'right lenticular nucleus': 147, 'thalami left occipital lobe brainstem and left cerebellum': 268, 'lentiform': 378, 'appendix': 342, 'anterior mediastinum': 86, 'double arch': 294, 'pleural effusion': 271, 'catheter': 400, 'gallstones': 39, 'ischemia': 52, 'periappendiceal fluid and fat stranding': 65, 'medical process': 124, 'trace gallbladder emptying': 416, 'with contrast': 184, 'mr flair': 173, 'pneumothorax': 438, 'proximal aspect of appendix': 356, 'posterior lung seen in image section': 437, 'left lateral aspect of anterior peritoneum': 142, 'medial and lateral rectus': 168, 'right lung base': 85, 'ascending colon': 283, 'adjacent to appendix': 357, 'right kidney': 258, 'biopsy': 373, 'there is massive cerebral hemisphere edema': 352, 'congenital developmental disorder history of surgery and past manipulation': 172, 'ct with gi and iv contrast': 159, 'ribs': 204, 'in right hilum': 95, 'less enhancement': 423, 'paratracheal area': 427, 'right lung': 44, 'portal vein occlusion': 350, '4th ventricle': 144, 'mri t2 weighted': 337, 'respiratory system': 151, 'right pca': 49, 'almost entire right side': 285, 'right subdural hematoma': 341, 'black': 297, 'infarcted areas': 110, 'left temporal lobe': 4, 'maybe': 412, 'abnormal hyperintensity in right occipital lobe': 365, 'short section irregular contour': 308, 'ring enhancing lesions': 37, 'necrosis': 370, 'it is enlarged with prominence of aortic knob': 445, 'less than half thorax': 58, 'flair mri': 141, 'central hyperintensity and surrounding hypointensity': 348, 'basal ganglia caudate and putamen': 431, 'hypointense': 319, 'enlarged fluid filled': 45, 'gray matter': 420, 'lateral film as well as pa': 217, 'single lung nodule': 234, 'radiolucent': 318, 'air': 148, 'soft tissue mass in region of terminal ileum with mesenteric lymphadenopathy': 343, 'fat': 123, 'right vertebral artery sign': 281, 'left upper lobe': 99, 'right sylvian fissure': 328, 'splenule': 369, 'left temporal horn': 224, 'left hepatic lobe': 22, 'diverticulitis': 245, '5mm': 300, 'necrotic tissue': 289, 'left occipital lobe': 158, 'midline': 292, 'oral and iv': 331, 'chest radiograph': 323, 'coronal': 135, 'mediport': 290, 'non contrast': 340, 'aorta is bright': 394, 'contrast ct with gi and iv contrast': 155, 'adenopathy': 425, 'bleeding in right posteroinferior cerebellum': 201, 'left': 82, 'bullous lesion': 92, 'axial': 10, '~15 minutes potentially faster with newer imaging systems': 122, 'blind loop syndrome': 236, 'just 1': 111, 'bilateral lungs': 309, 't2': 178, 'left lobe mass 1.5 x 1.8 cm': 406, 'acute stroke': 278, 'in bowel': 313, 'sinusitis': 382, 'left thalamus': 354, 'above clavicles bilaterally': 75, 'left apical pneumothorax': 136, 'non enhanced': 320, 'less dense': 73, 'calcification': 221, 'it is shifted to right': 105, 'increased opacity in left retrocardiac region': 443, 'solid': 115, 'mid abdomen': 322, 'posterior fossa': 327, 'genetic': 396, 'right convexity': 364, 'loss of normal gray white matter junction': 229, 'head neck ct': 372, 'thickening of bronchi': 389, 'heterogeneous': 193, 'metastases infection abcess glioblastoma': 374, 'pleural plaques': 415, 'middle mogul': 43, '5cm': 282, 'lung markings present all way laterally to ribs': 207, 'micronodular': 246, 'metastasis': 29, 'stomach': 311, 'mri diffusion weighted': 7, '1': 32, 'pericholecystic fluid': 261, 'diffusion weighted imaging dwi': 154, 'yes': 0, 'irregular': 452, 'smooth': 223, 'right paratracheal mass lesion': 444, 'hypoxic ischemic injury': 228, 'posteriorly': 371, 'hepatocellular carcioma': 344, 'cerebrum and lateral ventricles': 363, 'enhancement of vessels': 395, 'cardiomegaly': 138, 'moderate edema': 125, 'not seen here': 94, 'fluid': 286, 'bilateral frontal lobes': 239, 'cancer': 376, 'diffuse': 17, 'mri t1 weighted': 198, 'intestine': 210, 'mr adc map': 265, 'cardiopulmonary': 432, 'in bowels': 306, 'nodules': 266, 'haustra': 411, 'gastrointestinal': 76, 'head of pancreas': 140, 'spleen': 270, 'medial rectus': 163, 'basilar artery': 192, 'cavum vergae': 347, 'more acute means more inflammation leading to enhancement': 422, 'base': 332, 'vasculature': 6, 'right lower lateral lung field': 185, 'motion': 334, 'jejunum': 264, 'toxoplasma lymphoma abscesses other brain tumors': 375, 'pancreas': 19, 'scoliosis': 113, 'hemorrhage': 259, 'nothing': 353, 'upper lobes': 197, 'cardiomegaly with pulmonary edema': 130, 'ct': 18, 'viral': 383, 'r frontal lobe': 175, 'right of midline superior to right hilum': 315, 'punctate': 345, '12': 454, 'in cortex and basal ganglia bilaterally': 405, 'mri dwi': 240, 'ascites': 257, '0': 59, 'infiltrative': 453, 'in midline': 316, 'pituitary fossa': 408, 'diverticuli': 335, 'right mainstem bronchus is more in line with trachea than left': 208, 'cxr': 134, 'anterior to transverse colon': 126, 'right superior cavoatrial junction': 51, 'sigmoid flexture of colon': 325}
# json.dump(ans2label_dic, open("/root/VQA_Main/Modified_MedVQA-main/data/ans2label.json",'w'))
# ans2label = json.load(open("/root/VQA_Main/Modified_MedVQA-main/data/ans2label.json",'r'))
# print(len(ans2label))
# ans2label_all_answers = [k for k,v in ans2label_dic.items()]
# print(len(ans2label_all_answers))
# ans2label_all_answers = list(set(ans2label_all_answers))
# print(len(ans2label_all_answers))

# opened_answer = [str(a).lower() for a in opened_answer]
# ans2label_opened_answers = [a for a in opened_answer if a in set(ans2label_all_answers) ]
# print("opened answer number using MMQ",len(ans2label_opened_answers))

#=======dump to json====
# ans2label_opened = {ans:idx for idx, ans in enumerate(ans2label_opened_answers)}
# json.dump(ans2idx_opened, open('/root/VQA_Main/Modified_MedVQA-main/data/ans2label_opened.json', 'w'))

#==================create data using json files
train = json.load(open('/root/VQA_Main/Modified_MedVQA-main/data/trainset.json', 'r'))
test = json.load(open('/root/VQA_Main/Modified_MedVQA-main/data/testset.json', 'r'))
data = train + test
#====================get answer label which using for classifcation labels=========
ans2label_all_answers = json.load(open('/root/VQA_Main/Modified_MedVQA-main/data/ans2label.json', 'r'))
print("+========================len of all answers labels using for classifcation================", len(ans2label_all_answers))
#==================create closed question pool
# closed_answers = json.load(open('/root/VQA_Main/Modified_MedVQA-main/data/ans2idx_closed.json', 'r'))
closed_answers = [i["answer"] for i in data if 'CLOSED' in i['answer_type']]
print(type(closed_answers))
for i in range(len(closed_answers)):
    closed_answers[i] = str(closed_answers[i]).lower().replace(',', '').replace('?', '').replace('\'s', ' \'s').replace('...', '').replace('x ray', 'x-ray').replace('.', '')
ans2label_closed_answers = [a for a in closed_answers if a in set(ans2label_all_answers) ]
closed_answers = list(set(closed_answers))
print("+========================len of closed answers================", len(closed_answers))

print(len(ans2label_closed_answers))
ans2label_closed = {ans:idx for idx, ans in enumerate(closed_answers)}
json.dump(ans2label_closed, open('/root/VQA_Main/Modified_MedVQA-main/data/ans2label_closeded.json', 'w'))
print(len(ans2label_closed))

#==================create opened question pool
# opened_answers = json.load(open('/root/VQA_Main/Modified_MedVQA-main/data/ans2idx_opened.json', 'r'))
opened_answers = [i["answer"] for i in data if i['answer_type'] == 'OPEN']

for i in range(len(opened_answers)):
    opened_answers[i] = str(opened_answers[i]).lower().replace(',', '').replace('?', '').replace('\'s', ' \'s').replace('...', '').replace('x ray', 'x-ray').replace('.', '').replace('-', ' ')

opened_answers = list(set(opened_answers)) #515
opened_answers = [a for a in opened_answers if a in set(ans2label_all_answers) ]
print("+========================len of opened answers after filter================",len(opened_answers))
ans2label_opened = {ans:idx for idx, ans in enumerate(opened_answers)}
json.dump(ans2label_opened, open('/root/VQA_Main/Modified_MedVQA-main/data/ans2label_opened.json', 'w'))
print(len(ans2label_opened))

#===================================Deal with datas==============
train = json.load(open('/root/VQA_Main/Modified_MedVQA-main/data/trainset.json', 'r'))
test = json.load(open('/root/VQA_Main/Modified_MedVQA-main/data/testset.json', 'r'))
for i in train:
    i['answer'] = str(i['answer']).lower().replace(',', '').replace('?', '').replace('\'s', ' \'s').replace('...', '').replace('x ray', 'x-ray').replace('.', '').replace('-', ' ')
for i in test:
    i['answer'] = str(i['answer']).lower().replace(',', '').replace('?', '').replace('\'s', ' \'s').replace('...', '').replace('x ray', 'x-ray').replace('.', '').replace('-', ' ')
train_set = [a for a in train if a['answer'] in set(ans2label_all_answers) ]
test_set = [a for a in test if a['answer'] in set(ans2label_all_answers) ]

annotation = {'train': train_set, 'test': test_set}


json.dump(annotation, open('/root/VQA_Main/Modified_MedVQA-main/data/annotation.json', 'w'))

#============================all answer in json file==================
# train = json.load(open('/root/VQA_Main/Modified_MedVQA-main/data/trainset.json', 'r'))
# test = json.load(open('/root/VQA_Main/Modified_MedVQA-main/data/testset.json', 'r'))
# data = train + test
for i in data:
    i['answer'] = str(i['answer']).lower().replace(',', '').replace('?', '').replace('\'s', ' \'s').replace('...', '').replace('x ray', 'x-ray').replace('.', '').replace('-', ' ')

answer = [i['answer'] for i in data]
answer = list(set(answer))
print('=======length of all answer in data', len(answer))