import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LambdaLR
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import random
import itertools 
import torch.nn.init as init
from sklearn.model_selection import train_test_split



# --- Klasör Yolları ---
afm_folder_path = r'C:\Python\O2A Dataset\AFM'
o2a_folder_path = r'C:\Python\O2A Dataset\O2A'



IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.tiff', '.bmp')

def count_images_in_folder(folder_path):
    """Verilen klasör yolundaki geçerli görüntü dosyalarının sayısını döndürür."""
    if not os.path.isdir(folder_path):
        print(f"UYARI: Klasör bulunamadı: {folder_path}")
        return 0
    
    image_count = 0
    
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(IMAGE_EXTENSIONS):
            image_count += 1
            
    return image_count


afm_count = count_images_in_folder(afm_folder_path)
o2a_count = count_images_in_folder(o2a_folder_path)

total_count = afm_count + o2a_count

print("-" * 30)
print(f"AFM Klasöründeki Görüntü Sayısı: {afm_count}")
print(f"O2A Klasöründeki Görüntü Sayısı: {o2a_count}")
print("-" * 30)
print(f"Toplam Görüntü Sayısı: {total_count}")



num_epochs = 200
lr = 0.0002 # CycleGAN LR
batch_size = 8 
LAMBDA_CYCLE = 10.0 
LAMBDA_IDENTITY = 0.5 
IMAGE_SIZE = 128
BETA1 = 0.5



checkpoint_dir = "./cyclegan_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)



device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Kullanılan Cihaz: {device}")



import sys, torch
print("python:", sys.executable)
print("python version:", sys.version.replace('\\n',' '))
print("torch:", getattr(torch, '__version__', 'not installed'))
print("cuda available (before):", torch.cuda.is_available() if 'torch' in sys.modules else 'torch not imported yet')



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)



def save_checkpoint(G_A2B, G_B2A, D_A, D_B, opt_G, opt_D, epoch):
    torch.save({
        'epoch': epoch,
        'G_A2B_state_dict': G_A2B.state_dict(),
        'G_B2A_state_dict': G_B2A.state_dict(),
        'D_A_state_dict': D_A.state_dict(),
        'D_B_state_dict': D_B.state_dict(),
        'opt_G_state_dict': opt_G.state_dict(),
        'opt_D_state_dict': opt_D.state_dict()
    }, os.path.join(checkpoint_dir, f'cyclegan_checkpoint_epoch_{epoch}.pth'))
    print(f"📦 CycleGAN checkpoint saved at epoch {epoch}.")

def load_checkpoint(path, G_A2B, G_B2A, D_A, D_B, opt_G, opt_D):
    checkpoint = torch.load(path)
    G_A2B.load_state_dict(checkpoint['G_A2B_state_dict'])
    G_B2A.load_state_dict(checkpoint['G_B2A_state_dict'])
    D_A.load_state_dict(checkpoint['D_A_state_dict'])
    D_B.load_state_dict(checkpoint['D_B_state_dict'])
    opt_G.load_state_dict(checkpoint['opt_G_state_dict'])
    opt_D.load_state_dict(checkpoint['opt_D_state_dict'])
    print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
    return checkpoint['epoch']



def weights_init_normal(m):
    classname = m.__class__.__name__
    
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)
            
    elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)




def initialize_image_buffer(max_size=50):
    return {'data': [], 'max_size': max_size}

def push_and_pop_image(buffer, data):
    to_return = []
    
    for element in data:
        element = torch.unsqueeze(element, 0)
        
        if len(buffer['data']) < buffer['max_size']:
            buffer['data'].append(element)
            to_return.append(element)
        else:
            if random.uniform(0, 1) > 0.5:
                i = random.randint(0, buffer['max_size'] - 1)
                to_return.append(buffer['data'][i].clone())
                buffer['data'][i] = element
            else:
                to_return.append(element)
                
    return torch.cat(to_return, 0)



def create_residual_block(in_features):
    return nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_features, in_features, 3),
        nn.BatchNorm2d(in_features),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_features, in_features, 3),
        nn.BatchNorm2d(in_features)
    )

def create_generator_resnet(input_nc=3, output_nc=3, num_residual_blocks=9):
    model = [
        nn.ReflectionPad2d(3), 
        nn.Conv2d(input_nc, 64, 7), 
        nn.BatchNorm2d(64), 
        nn.ReLU(inplace=True)
    ]
    
    in_features = 64; out_features = in_features * 2
    for _ in range(2): 
        model += [
            nn.Conv2d(in_features, out_features, 3, stride=2, padding=1), 
            nn.BatchNorm2d(out_features), 
            nn.ReLU(inplace=True)
        ]
        in_features = out_features; out_features = in_features * 2
        
    
    
    for _ in range(num_residual_blocks): 
         model += [create_residual_block(in_features)] 
         
    out_features = in_features // 2
    for _ in range(2): 
        model += [
            nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1), 
            nn.BatchNorm2d(out_features), 
            nn.ReLU(inplace=True)
        ]
        in_features = out_features; out_features = in_features // 2
        
    model += [
        nn.ReflectionPad2d(3), 
        nn.Conv2d(64, output_nc, 7), 
        nn.Tanh()
    ]
    
    return nn.Sequential(*model)

def create_discriminator(input_nc=3):
    model = [
        nn.Conv2d(input_nc, 64, 4, stride=2, padding=1), 
        nn.LeakyReLU(0.2, inplace=True)
    ]
    model += [
        nn.Conv2d(64, 128, 4, stride=2, padding=1), 
        nn.BatchNorm2d(128), 
        nn.LeakyReLU(0.2, inplace=True)
    ]
    model += [
        nn.Conv2d(128, 256, 4, stride=2, padding=1), 
        nn.BatchNorm2d(256), 
        nn.LeakyReLU(0.2, inplace=True)
    ]
    model += [
        nn.Conv2d(256, 512, 4, padding=1), 
        nn.BatchNorm2d(512), 
        nn.LeakyReLU(0.2, inplace=True)
    ]
    model += [
        nn.Conv2d(512, 1, 4, padding=1)
    ]
    
    return nn.Sequential(*model)




def cyclegan_data_generator(afm_files, o2a_files, root_A, root_B, transform):
    len_A = len(afm_files)
    len_B = len(o2a_files)
    max_len = max(len_A, len_B)
    
    for index in range(max_len):
        path_A = os.path.join(root_A, afm_files[index % len_A])
        path_B = os.path.join(root_B, o2a_files[random.randint(0, len_B - 1)])

        img_A = Image.open(path_A).convert('RGB')
        img_B = Image.open(path_B).convert('RGB')
        
        yield transform(img_A), transform(img_B)



def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image

def load_cyclegan_images(afm_files, o2a_files, afm_folder_path, o2a_folder_path, transform):
    len_A = len(afm_files)
    len_B = len(o2a_files)
    max_len = max(len_A, len_B)
    
    for index_A in range(max_len):
        
        afm_name = afm_files[index_A % len_A]
        afm_image = load_image(os.path.join(afm_folder_path, afm_name))
        
        index_B = random.randint(0, len_B - 1)
        o2a_name = o2a_files[index_B]
        o2a_image = load_image(os.path.join(o2a_folder_path, o2a_name))
        
        afm_image = transform(afm_image)
        o2a_image = transform(o2a_image)

        yield afm_image, o2a_image


def create_cyclegan_dataloader_from_lists(afm_files, o2a_files, afm_folder_path, o2a_folder_path, batch_size=1, shuffle=False):
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = list(load_cyclegan_images(
        afm_files, 
        o2a_files, 
        afm_folder_path, 
        o2a_folder_path, 
        transform
    ))
    
    NUM_WORKERS = 0 
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        
        num_workers=NUM_WORKERS, 
        pin_memory=True             
    )
    return dataloader



def train_cyclegan(G_A2B, G_B2A, D_A, D_B, dataloader, 
                   optimizer_G, optimizer_D, num_epochs,
                   val_dataloader=None, device='cuda', save_every=10, resume_checkpoint=None):

    G_A2B.to(device); G_B2A.to(device)
    D_A.to(device); D_B.to(device)
    
    criterion_GAN = nn.MSELoss() 
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    
    fake_A_buffer_dict = initialize_image_buffer()
    fake_B_buffer_dict = initialize_image_buffer()
    
    LAMBDA_CYCLE = 10.0
    LAMBDA_IDENTITY = 0.5
    
    def lambda_lr(epoch):
        decay_start_epoch = 100
        if epoch < decay_start_epoch:
            return 1.0
        else:
            decay_period = num_epochs - decay_start_epoch
            if decay_period <= 0: return 1.0
            progress = epoch - decay_start_epoch
            return 1.0 - (progress / decay_period)
            
    scheduler_G = LambdaLR(optimizer_G, lr_lambda=lambda_lr)
    scheduler_D = LambdaLR(optimizer_D, lr_lambda=lambda_lr)
    
    start_epoch = 0
    if resume_checkpoint:
        start_epoch = load_checkpoint(resume_checkpoint, G_A2B, G_B2A, D_A, D_B, optimizer_G, optimizer_D) 

    print("--- CycleGAN Eğitimi Başlıyor ---")
    
    for epoch in range(start_epoch, num_epochs):
        G_A2B.train(); G_B2A.train()
        D_A.train(); D_B.train()
        
        for i, (real_A, real_B) in enumerate(dataloader):
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            
            optimizer_G.zero_grad()
            
            loss_identity_B = criterion_identity(G_A2B(real_B), real_B) * LAMBDA_IDENTITY
            loss_identity_A = criterion_identity(G_B2A(real_A), real_A) * LAMBDA_IDENTITY
            fake_B = G_A2B(real_A)
            loss_GAN_A2B = criterion_GAN(D_B(fake_B), torch.ones_like(D_B(fake_B)).to(device))
            fake_A = G_B2A(real_B)
            loss_GAN_B2A = criterion_GAN(D_A(fake_A), torch.ones_like(D_A(fake_A)).to(device))
            recovered_A = G_B2A(fake_B)
            loss_cycle_A = criterion_cycle(recovered_A, real_A) * LAMBDA_CYCLE
            recovered_B = G_A2B(fake_A)
            loss_cycle_B = criterion_cycle(recovered_B, real_B) * LAMBDA_CYCLE
            loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_A + loss_cycle_B + loss_identity_A + loss_identity_B
            loss_G.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()
            
            loss_D_real_A = criterion_GAN(D_A(real_A), torch.ones_like(D_A(real_A)).to(device))
            fake_A_pooled = push_and_pop_image(fake_A_buffer_dict, fake_A.detach()) # Düzeltildi
            loss_D_fake_A = criterion_GAN(D_A(fake_A_pooled), torch.zeros_like(D_A(fake_A_pooled)).to(device))
            loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5
            loss_D_A.backward()
            
            loss_D_real_B = criterion_GAN(D_B(real_B), torch.ones_like(D_B(real_B)).to(device))
            fake_B_pooled = push_and_pop_image(fake_B_buffer_dict, fake_B.detach()) # Düzeltildi
            loss_D_fake_B = criterion_GAN(D_B(fake_B_pooled), torch.zeros_like(D_B(fake_B_pooled)).to(device))
            loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5
            loss_D_B.backward()
            optimizer_D.step()

            if i % 50 == 0:
                current_lr = optimizer_G.param_groups[0]['lr']
                loss_D = loss_D_A.item() + loss_D_B.item() 
                print(f"[Epoch {epoch+1}/{num_epochs}] [Batch {i+1}/{len(dataloader)}] "
                      f"[D Loss: {loss_D:.4f}] [G Loss: {loss_G.item():.4f}] (LR: {current_lr:.6f})")

        scheduler_G.step()
        scheduler_D.step()
        
        if val_dataloader is not None:
            pass 
            
        if (epoch + 1) % save_every == 0:
            save_checkpoint(G_A2B, G_B2A, D_A, D_B, optimizer_G, optimizer_D, epoch+1)
            
    print("🎉 CycleGAN Eğitimi Başarıyla Tamamlandı!")



def split_dataset_three_way(afm_folder_path, o2a_folder_path, test_ratio=0.1, val_ratio=0.1, random_state=42):
    
    afm_files = sorted(os.listdir(afm_folder_path))
    o2a_files = sorted(os.listdir(o2a_folder_path))

    
    train_afm_val_afm, test_afm, train_o2a_val_o2a, test_o2a = train_test_split(
        afm_files, o2a_files, test_size=test_ratio, random_state=random_state
    )

    
    val_size_adjusted = val_ratio / (1 - test_ratio)  

    train_afm, val_afm, train_o2a, val_o2a = train_test_split(
        train_afm_val_afm, train_o2a_val_o2a, test_size=val_size_adjusted, random_state=random_state
    )

    return (train_afm, train_o2a), (val_afm, val_o2a), (test_afm, test_o2a)



(train_afm, train_o2a), (val_afm, val_o2a), (test_afm, test_o2a) = split_dataset_three_way(
    afm_folder_path, 
    o2a_folder_path, 
    test_ratio=0.1, 
    val_ratio=0.1, 
    random_state=42
)



print(f"✅ Eğitim seti dosya sayısı (AFM): {len(train_afm)}")
print(f"✅ Doğrulama seti dosya sayısı (AFM): {len(val_afm)}")
print(f"✅ Test seti dosya sayısı (AFM): {len(test_afm)}")



# TRAIN LOADER
train_loader = create_cyclegan_dataloader_from_lists(
    afm_files=train_afm,         
    o2a_files=train_o2a,         
    afm_folder_path=afm_folder_path, 
    o2a_folder_path=o2a_folder_path, 
    batch_size=12, 
    shuffle=True
)

# VALIDATION LOADER
val_loader = create_cyclegan_dataloader_from_lists(
    afm_files=val_afm,
    o2a_files=val_o2a,
    afm_folder_path=afm_folder_path,
    o2a_folder_path=o2a_folder_path,
    batch_size=12, 
    shuffle=False
)

# TEST LOADER
test_loader = create_cyclegan_dataloader_from_lists(
    afm_files=test_afm,
    o2a_files=test_o2a,
    afm_folder_path=afm_folder_path,
    o2a_folder_path=o2a_folder_path,
    batch_size=12, 
    shuffle=False
)

print(f"✅ CycleGAN DataLoader'lar hazır. Train: {len(train_loader)} iter.")



G_A2B = create_generator_resnet().to(device) 
G_B2A = create_generator_resnet().to(device) 
D_A = create_discriminator().to(device)     
D_B = create_discriminator().to(device)     



G_A2B.apply(weights_init_normal)
G_B2A.apply(weights_init_normal)
D_A.apply(weights_init_normal)
D_B.apply(weights_init_normal)
print("✅ Modeller oluşturuldu ve ağırlıklar başlatıldı.")



optimizer_G = torch.optim.Adam(
    itertools.chain(G_A2B.parameters(), G_B2A.parameters()),
    lr=lr,
    betas=(BETA1, 0.999)
)
optimizer_D = torch.optim.Adam(
    itertools.chain(D_A.parameters(), D_B.parameters()),
    lr=lr,
    betas=(BETA1, 0.999)
)

print("✅ Modeller, Optimizer'lar ve Ağırlıklar Hazır.")



train_cyclegan(
    G_A2B, G_B2A, D_A, D_B, 
    dataloader=train_loader, 
    optimizer_G=optimizer_G, 
    optimizer_D=optimizer_D, 
    num_epochs=num_epochs,
    val_dataloader=val_loader, 
    device=device, 
    save_every=10
)


