import os
import json
import glob
from PIL import Image
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA

def main():
    out_dir = '/research/current_drive/agentic_coding/eigenfaces_demo'
    data_dir = os.path.join(out_dir, 'data')
    img_dir = os.path.join(data_dir, 'images')
    ef_dir = os.path.join(data_dir, 'eigenfaces')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ef_dir, exist_ok=True)

    print("1. Loading LFW training set (min_faces_per_person=70)...")
    lfw_train = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    X_train = lfw_train.data
    mean_face = np.mean(X_train, axis=0) # shape (1850,)

    n_components = 80
    print(f"2. Fitting PCA with {n_components} components on {X_train.shape[0]} images...")
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=False, random_state=42)
    pca.fit(X_train)

    cum_var = np.cumsum(pca.explained_variance_ratio_).tolist()
    eigenfaces = pca.components_ # (80, 1850)

    # Save Top 16 Eigenface PNGs
    print("3. Exporting eigenface preview PNGs...")
    for i in range(16):
        ef = eigenfaces[i].reshape(50, 37)
        ef_norm = (ef - ef.min()) / (ef.max() - ef.min() + 1e-8) * 255.0
        im = Image.fromarray(ef_norm.astype(np.uint8)).resize((111, 150), Image.NEAREST)
        im.save(os.path.join(ef_dir, f'eigenface_{i+1}.png'))

    # Save Mean Face PNG
    mf_img = (mean_face.reshape(50, 37) * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(mf_img).resize((111, 150), Image.NEAREST).save(os.path.join(ef_dir, 'mean_face.png'))

    # 4. Curate Diverse People
    print("4. Processing curated people across global demographics...")
    base_lfw = os.path.expanduser('~/scikit_learn_data/lfw_home/lfw_funneled')

    people_meta = [
        # Developed / In-Training Benchmark
        {
            "id": "george_w_bush",
            "name": "George W. Bush",
            "folder": "George_W_Bush",
            "img_idx": 0,
            "role": "43rd U.S. President",
            "country": "United States",
            "flag": "🇺🇸",
            "region": "North America",
            "gender": "Male",
            "demographic": "White Male (Western Leader)",
            "inTrainingSet": True,
            "trainingImagesCount": 530,
            "bio": "43rd President of the United States. Comprises over 41% of the scikit-learn training set."
        },
        {
            "id": "tony_blair",
            "name": "Tony Blair",
            "folder": "Tony_Blair",
            "img_idx": 0,
            "role": "UK Prime Minister (1997–2007)",
            "country": "United Kingdom",
            "flag": "🇬🇧",
            "region": "Europe",
            "gender": "Male",
            "demographic": "White Male (Western Leader)",
            "inTrainingSet": True,
            "trainingImagesCount": 144,
            "bio": "British Prime Minister. Second most photographed European leader in LFW."
        },
        {
            "id": "colin_powell",
            "name": "Colin Powell",
            "folder": "Colin_Powell",
            "img_idx": 0,
            "role": "65th U.S. Secretary of State",
            "country": "United States",
            "flag": "🇺🇸",
            "region": "North America",
            "gender": "Male",
            "demographic": "Black Male (Western Diplomat)",
            "inTrainingSet": True,
            "trainingImagesCount": 236,
            "bio": "First African-American Secretary of State and Chairman of the Joint Chiefs of Staff."
        },
        {
            "id": "gerhard_schroeder",
            "name": "Gerhard Schröder",
            "folder": "Gerhard_Schroeder",
            "img_idx": 0,
            "role": "Chancellor of Germany (1998–2005)",
            "country": "Germany",
            "flag": "🇩🇪",
            "region": "Europe",
            "gender": "Male",
            "demographic": "White Male (Western Leader)",
            "inTrainingSet": True,
            "trainingImagesCount": 109,
            "bio": "German Chancellor and European political leader heavily represented in LFW."
        },
        {
            "id": "donald_rumsfeld",
            "name": "Donald Rumsfeld",
            "folder": "Donald_Rumsfeld",
            "img_idx": 0,
            "role": "U.S. Secretary of Defense",
            "country": "United States",
            "flag": "🇺🇸",
            "region": "North America",
            "gender": "Male",
            "demographic": "White Male (Western Leader)",
            "inTrainingSet": True,
            "trainingImagesCount": 121,
            "bio": "U.S. Secretary of Defense during the 2002–2003 Iraq war news cycle."
        },
        {
            "id": "hugo_chavez",
            "name": "Hugo Chávez",
            "folder": "Hugo_Chavez",
            "img_idx": 0,
            "role": "President of Venezuela (1999–2013)",
            "country": "Venezuela",
            "flag": "🇻🇪",
            "region": "Latin America",
            "gender": "Male",
            "demographic": "Mestizo Male (Latin American Leader)",
            "inTrainingSet": True,
            "trainingImagesCount": 71,
            "bio": "Venezuelan President, the only Latin American leader with ≥70 images in LFW."
        },

        # Global Leaders & Developing Nations (Out of Training Set)
        {
            "id": "nelson_mandela",
            "name": "Nelson Mandela",
            "folder": "Nelson_Mandela",
            "img_idx": 0,
            "role": "President of South Africa (1994–1999)",
            "country": "South Africa",
            "flag": "🇿🇦",
            "region": "Africa",
            "gender": "Male",
            "demographic": "Black Male (African Leader & Nobel Laureate)",
            "inTrainingSet": False,
            "trainingImagesCount": 4,
            "bio": "Anti-apartheid revolutionary, global icon of peace, and Nobel Peace Prize laureate."
        },
        {
            "id": "kofi_annan",
            "name": "Kofi Annan",
            "folder": "Kofi_Annan",
            "img_idx": 0,
            "role": "7th UN Secretary-General",
            "country": "Ghana",
            "flag": "🇬🇭",
            "region": "Africa",
            "gender": "Male",
            "demographic": "Black Male (African Diplomat & Nobel Laureate)",
            "inTrainingSet": False,
            "trainingImagesCount": 32,
            "bio": "Ghanaian diplomat, 7th Secretary-General of the United Nations, and Nobel Peace Prize laureate."
        },
        {
            "id": "hu_jintao",
            "name": "Hu Jintao",
            "folder": "Hu_Jintao",
            "img_idx": 0,
            "role": "President of China (2003–2013)",
            "country": "China",
            "flag": "🇨🇳",
            "region": "Asia",
            "gender": "Male",
            "demographic": "East Asian Male (Asian Leader)",
            "inTrainingSet": False,
            "trainingImagesCount": 15,
            "bio": "General Secretary of the Chinese Communist Party and President of China."
        },
        {
            "id": "junichiro_koizumi",
            "name": "Junichiro Koizumi",
            "folder": "Junichiro_Koizumi",
            "img_idx": 0,
            "role": "Prime Minister of Japan (2001–2006)",
            "country": "Japan",
            "flag": "🇯🇵",
            "region": "Asia",
            "gender": "Male",
            "demographic": "East Asian Male (Asian Leader)",
            "inTrainingSet": False,
            "trainingImagesCount": 60,
            "bio": "Japanese Prime Minister who was prominent in international diplomacy during 2001–2006."
        },
        {
            "id": "megawati_sukarnoputri",
            "name": "Megawati Sukarnoputri",
            "folder": "Megawati_Sukarnoputri",
            "img_idx": 0,
            "role": "5th President of Indonesia (2001–2004)",
            "country": "Indonesia",
            "flag": "🇮🇩",
            "region": "Asia",
            "gender": "Female",
            "demographic": "Southeast Asian Female (Asian Leader)",
            "inTrainingSet": False,
            "trainingImagesCount": 33,
            "bio": "First female President of Indonesia and leader of the world's most populous Muslim-majority nation."
        },
        {
            "id": "mahathir_mohamad",
            "name": "Mahathir Mohamad",
            "folder": "Mahathir_Mohamad",
            "img_idx": 0,
            "role": "Prime Minister of Malaysia",
            "country": "Malaysia",
            "flag": "🇲🇾",
            "region": "Asia",
            "gender": "Male",
            "demographic": "Southeast Asian Male (Asian Leader)",
            "inTrainingSet": False,
            "trainingImagesCount": 14,
            "bio": "Architect of Malaysia's modern economic modernization and outspoken Global South leader."
        },
        {
            "id": "atal_bihari_vajpayee",
            "name": "Atal Bihari Vajpayee",
            "folder": "Atal_Bihari_Vajpayee",
            "img_idx": 0,
            "role": "Prime Minister of India (1998–2004)",
            "country": "India",
            "flag": "🇮🇳",
            "region": "Asia",
            "gender": "Male",
            "demographic": "South Asian Male (Asian Leader)",
            "inTrainingSet": False,
            "trainingImagesCount": 24,
            "bio": "10th Prime Minister of India, statesman, poet, and leader of the world's largest democracy."
        },
        {
            "id": "luiz_inacio_lula_da_silva",
            "name": "Luiz Inácio Lula da Silva",
            "folder": "Luiz_Inacio_Lula_da_Silva",
            "img_idx": 0,
            "role": "President of Brazil",
            "country": "Brazil",
            "flag": "🇧🇷",
            "region": "Latin America",
            "gender": "Male",
            "demographic": "Latin American Male (Global South Leader)",
            "inTrainingSet": False,
            "trainingImagesCount": 48,
            "bio": "Prominent Brazilian trade union leader and President of Latin America's largest nation."
        },
        {
            "id": "vicente_fox",
            "name": "Vicente Fox",
            "folder": "Vicente_Fox",
            "img_idx": 0,
            "role": "President of Mexico (2000–2006)",
            "country": "Mexico",
            "flag": "🇲🇽",
            "region": "Latin America",
            "gender": "Male",
            "demographic": "Hispanic Male (Latin American Leader)",
            "inTrainingSet": False,
            "trainingImagesCount": 32,
            "bio": "55th President of Mexico, whose historic 2000 election ended 71 years of PRI rule."
        },
        {
            "id": "hamid_karzai",
            "name": "Hamid Karzai",
            "folder": "Hamid_Karzai",
            "img_idx": 0,
            "role": "President of Afghanistan (2001–2014)",
            "country": "Afghanistan",
            "flag": "🇦🇫",
            "region": "Asia",
            "gender": "Male",
            "demographic": "Central Asian / Middle Eastern Male",
            "inTrainingSet": False,
            "trainingImagesCount": 22,
            "bio": "First democratically elected President of Afghanistan, known for his traditional karakul hat and chapan."
        },

        # Women & Cultural / Sports Icons
        {
            "id": "serena_williams",
            "name": "Serena Williams",
            "folder": "Serena_Williams",
            "img_idx": 0,
            "role": "Tennis Legend (23 Grand Slams)",
            "country": "United States",
            "flag": "🇺🇸",
            "region": "North America",
            "gender": "Female",
            "demographic": "Black Female (World Champion Athlete)",
            "inTrainingSet": False,
            "trainingImagesCount": 52,
            "bio": "Widely regarded as one of the greatest athletes of all time, with 23 Grand Slam singles titles."
        },
        {
            "id": "gloria_macapagal_arroyo",
            "name": "Gloria Macapagal-Arroyo",
            "folder": "Gloria_Macapagal_Arroyo",
            "img_idx": 0,
            "role": "President of the Philippines (2001–2010)",
            "country": "Philippines",
            "flag": "🇵🇭",
            "region": "Asia",
            "gender": "Female",
            "demographic": "Southeast Asian Female (Asian Leader)",
            "inTrainingSet": False,
            "trainingImagesCount": 44,
            "bio": "14th President of the Philippines, economist, and second female Philippine president."
        },
        {
            "id": "salma_hayek",
            "name": "Salma Hayek",
            "folder": "Salma_Hayek",
            "img_idx": 0,
            "role": "Actress & Producer",
            "country": "Mexico",
            "flag": "🇲🇽",
            "region": "Latin America",
            "gender": "Female",
            "demographic": "Hispanic Female (Cultural Icon)",
            "inTrainingSet": False,
            "trainingImagesCount": 13,
            "bio": "Mexican and American actress, Academy Award nominee for Frida, and international cultural figure."
        },
        {
            "id": "jackie_chan",
            "name": "Jackie Chan",
            "folder": "Jackie_Chan",
            "img_idx": 0,
            "role": "Martial Artist & Actor",
            "country": "Hong Kong",
            "flag": "🇭🇰",
            "region": "Asia",
            "gender": "Male",
            "demographic": "East Asian Male (Global Cultural Icon)",
            "inTrainingSet": False,
            "trainingImagesCount": 13,
            "bio": "Hong Kong martial artist, actor, and filmmaker known for his acrobatic fighting style and worldwide fame."
        },
        {
            "id": "condoleezza_rice",
            "name": "Condoleezza Rice",
            "folder": "Condoleezza_Rice",
            "img_idx": 0,
            "role": "66th U.S. Secretary of State",
            "country": "United States",
            "flag": "🇺🇸",
            "region": "North America",
            "gender": "Female",
            "demographic": "Black Female (Diplomat & Academic)",
            "inTrainingSet": False,
            "trainingImagesCount": 11,
            "bio": "First African-American female Secretary of State and first female National Security Advisor."
        },
        {
            "id": "jennifer_aniston",
            "name": "Jennifer Aniston",
            "folder": "Jennifer_Aniston",
            "img_idx": 0,
            "role": "Actress & Producer",
            "country": "United States",
            "flag": "🇺🇸",
            "region": "North America",
            "gender": "Female",
            "demographic": "White Female (Cultural Icon)",
            "inTrainingSet": False,
            "trainingImagesCount": 21,
            "bio": "American actress and global pop culture icon, famous for Friends and Hollywood cinema."
        },
        {
            "id": "arnold_schwarzenegger",
            "name": "Arnold Schwarzenegger",
            "folder": "Arnold_Schwarzenegger",
            "img_idx": 0,
            "role": "Governor of California (2003–2011)",
            "country": "Austria / USA",
            "flag": "🇦🇹",
            "region": "North America",
            "gender": "Male",
            "demographic": "White Male (Cultural & Political Icon)",
            "inTrainingSet": False,
            "trainingImagesCount": 42,
            "bio": "Austrian-American bodybuilder, Hollywood superstar, and 38th Governor of California."
        },
        {
            "id": "bill_clinton",
            "name": "Bill Clinton",
            "folder": "Bill_Clinton",
            "img_idx": 0,
            "role": "42nd U.S. President",
            "country": "United States",
            "flag": "🇺🇸",
            "region": "North America",
            "gender": "Male",
            "demographic": "White Male (Western Leader)",
            "inTrainingSet": False,
            "trainingImagesCount": 29,
            "bio": "42nd President of the United States who presided over the longest peacetime economic expansion."
        }
    ]

    processed_people = []

    for person in people_meta:
        folder_path = os.path.join(base_lfw, person['folder'])
        files = sorted(glob.glob(os.path.join(folder_path, '*.jpg')))
        if not files:
            print(f"Warning: No images found for {person['name']}")
            continue

        img_file = files[min(person['img_idx'], len(files)-1)]

        # Process exactly like scikit-learn
        with Image.open(img_file) as img:
            # Crop to face slice (78, 70, 172, 195)
            img_crop = img.crop((78, 70, 172, 195))
            img_resized = img_crop.resize((37, 50))

            # Save clean cropped photo for avatar and original display (scaled up to 148x200)
            img_display = img_crop.resize((148, 200), Image.BICUBIC)
            img_filename = f"{person['id']}.png"
            img_display.save(os.path.join(img_dir, img_filename))

            # Normalized float array
            arr = np.asarray(img_resized, dtype=np.float32) / 255.0
            if arr.ndim == 3:
                arr = arr.mean(axis=2)
            pixels = arr.flatten()

        # Project onto eigenfaces
        weights = np.dot(pixels - mean_face, eigenfaces.T) # shape (80,)

        # Compute MSE curve at K = [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80]
        k_steps = [1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80]
        mse_curve = {}
        for k in k_steps:
            recon = mean_face + np.dot(weights[:k], eigenfaces[:k])
            mse_val = float(np.mean((pixels - recon)**2))
            mse_curve[str(k)] = round(mse_val, 6)

        processed_people.append({
            "id": person["id"],
            "name": person["name"],
            "role": person["role"],
            "country": person["country"],
            "flag": person["flag"],
            "region": person["region"],
            "gender": person["gender"],
            "demographic": person["demographic"],
            "inTrainingSet": person["inTrainingSet"],
            "trainingImagesCount": person["trainingImagesCount"],
            "bio": person["bio"],
            "image": f"data/images/{img_filename}",
            "pixels": [round(float(p), 3) for p in pixels],
            "weights": [round(float(w), 3) for w in weights],
            "mse_curve": mse_curve
        })
        print(f"Processed: {person['name']} (MSE@K=20: {mse_curve['20']})")

    # 5. Pack everything into faces_data.json
    dataset = {
        "metadata": {
            "title": "LFW Eigenfaces Dataset for PCA Reconstruction",
            "n_pixels": 1850,
            "height": 50,
            "width": 37,
            "n_components": n_components,
            "training_samples": int(X_train.shape[0]),
            "demographic_skew": {
                "training_set": {
                    "total_images": 1288,
                    "male_percentage": 100.0,
                    "female_percentage": 0.0,
                    "white_western_percentage": 88.5,
                    "george_w_bush_share": 41.1,
                    "developing_nations_share": 5.5
                },
                "full_lfw_benchmark": {
                    "male_percentage": 77.5,
                    "female_percentage": 22.5,
                    "caucasian_percentage": 83.5,
                    "asian_percentage": 8.7,
                    "black_percentage": 4.5,
                    "hispanic_percentage": 2.8,
                    "us_europe_geography": 76.8
                }
            }
        },
        "mean_face": [round(float(p), 3) for p in mean_face],
        "variance_ratio": [round(float(v), 5) for v in pca.explained_variance_ratio_],
        "cumulative_variance": [round(float(c), 5) for c in cum_var],
        "eigenfaces": [[round(float(val), 4) for val in ef] for ef in eigenfaces],
        "people": processed_people
    }

    json_path = os.path.join(data_dir, 'faces_data.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, separators=(',', ':'))

    size_mb = os.path.getsize(json_path) / (1024 * 1024)
    print(f"\nSUCCESS! Generated faces_data.json ({size_mb:.2f} MB) with {len(processed_people)} people.")

if __name__ == '__main__':
    main()
