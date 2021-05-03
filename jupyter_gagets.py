# %%
import os
import pandas as pd

# %%
associ_csv = './Dataset/instruction/driver_imgs_list.csv'
df = pd.read_csv(associ_csv)

# %% extract subjects
print("Totally contains {} subjects".format(len(df['subject'].unique())))
val_subjects = ['p035', 'p072', 'p052', 'p081', 'p045']
train_subjects = []
for i in df['subject'].unique():
    if i not in val_subjects:
        train_subjects.append(i)
print(f"train_subjects: {len(train_subjects)} val_subjects {len(val_subjects)}")
# %% get the splits instruction
print(f"Totally contains samples: {len(df)}")
def leave_subject_out_instruction(ins_train, ins_val, val_subjects, img_root='../data/statefarm/imgs/train/'):
    train_writer = open(ins_train, 'w')
    val_writer = open(ins_val, 'w')

    for index, row in df.iterrows():
        sub_folder = row['classname']
        img_id = row['img']
        msg = os.path.join(img_root, sub_folder, img_id)
        msg += ',' + sub_folder[-1] + ',' + row['subject'] + '\n'
        if row['subject'] in val_subjects:
            val_writer.write(msg)
        else:
            train_writer.write(msg)
    print("Done")

ins_train_file = './Dataset/instruction/statefarm_train.txt'
ins_val_file = './Dataset/instruction/statefarm_val.txt'
leave_subject_out_instruction(ins_train_file, ins_val_file, val_subjects)
    





# %%
