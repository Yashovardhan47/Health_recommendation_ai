"""
Generate all synthetic CSV datasets for HealthAI
Run once: python data/generate_data.py
"""
import os, random, numpy as np, pandas as pd
from pathlib import Path

random.seed(42); np.random.seed(42)
HERE = Path(__file__).parent

DISEASES = [
    ("D001","Common Cold","Infectious","Mild"),
    ("D002","Influenza","Infectious","Moderate"),
    ("D003","COVID-19","Infectious","Moderate"),
    ("D004","Diabetes Type 2","Metabolic","Chronic"),
    ("D005","Hypertension","Cardiovascular","Chronic"),
    ("D006","Migraine","Neurological","Moderate"),
    ("D007","Asthma","Respiratory","Chronic"),
    ("D008","Pneumonia","Infectious","Severe"),
    ("D009","Gastroenteritis","Gastrointestinal","Mild"),
    ("D010","UTI","Urological","Mild"),
    ("D011","Anemia","Hematological","Moderate"),
    ("D012","Arthritis","Musculoskeletal","Chronic"),
    ("D013","Depression","Psychiatric","Chronic"),
    ("D014","Anxiety Disorder","Psychiatric","Moderate"),
    ("D015","Allergic Rhinitis","Immunological","Mild"),
]

MEDICINES = [
    ("M001","Cetirizine","Antihistamine","D001",10,0.82,120,"10mg once daily"),
    ("M002","Paracetamol","Analgesic","D001,D002,D006",500,0.88,30,"500-1000mg q6h"),
    ("M003","Pseudoephedrine","Decongestant","D001,D015",60,0.79,45,"60mg twice daily"),
    ("M004","Oseltamivir","Antiviral","D002,D003",75,0.85,280,"75mg twice daily x5d"),
    ("M005","Remdesivir","Antiviral","D003",200,0.80,5000,"200mg IV day1"),
    ("M006","Metformin","Antidiabetic","D004",500,0.88,35,"500-2000mg daily"),
    ("M007","Glipizide","Sulfonylurea","D004",5,0.82,55,"5-10mg daily"),
    ("M008","Amlodipine","CCB","D005",5,0.87,60,"5-10mg daily"),
    ("M009","Lisinopril","ACE Inhibitor","D005",10,0.89,45,"10-40mg daily"),
    ("M010","Sumatriptan","Triptan","D006",50,0.90,420,"50mg at onset"),
    ("M011","Topiramate","Anticonvulsant","D006",25,0.78,180,"25-100mg daily"),
    ("M012","Salbutamol","SABA","D007",100,0.92,180,"100-200mcg PRN"),
    ("M013","Fluticasone Inhaler","ICS","D007",250,0.88,620,"250mcg twice daily"),
    ("M014","Amoxicillin","Antibiotic","D008,D009",500,0.84,95,"500mg TDS x7d"),
    ("M015","Azithromycin","Macrolide","D008",500,0.83,180,"500mg day1 then 250mg x4d"),
    ("M016","ORS","Electrolyte","D009,D010",20,0.90,15,"200ml after each stool"),
    ("M017","Nitrofurantoin","Antibiotic","D010",100,0.87,140,"100mg twice daily x5d"),
    ("M018","Ferrous Sulfate","Iron Supplement","D011",325,0.88,40,"325mg twice daily"),
    ("M019","Folic Acid","Vitamin","D011,D013",5,0.90,25,"5mg daily"),
    ("M020","Celecoxib","COX-2 Inhibitor","D012",200,0.82,350,"200mg daily"),
    ("M021","Methotrexate","DMARD","D012",10,0.85,180,"7.5-25mg weekly"),
    ("M022","Sertraline","SSRI","D013,D014",50,0.83,145,"50-200mg daily"),
    ("M023","Escitalopram","SSRI","D013,D014",10,0.85,220,"10-20mg daily"),
    ("M024","Bupropion","NDRI","D013",150,0.81,380,"150-300mg daily"),
    ("M025","Alprazolam","Benzodiazepine","D014",0.25,0.89,75,"0.25-0.5mg TDS"),
    ("M026","Buspirone","Anxiolytic","D014",5,0.77,165,"5-20mg TDS"),
    ("M027","Fluticasone Nasal","INCS","D015",50,0.91,320,"1-2 sprays daily"),
    ("M028","Loratadine","Antihistamine","D015,D001",10,0.82,65,"10mg daily"),
    ("M029","Montelukast","LTRA","D015,D007",10,0.78,350,"10mg nightly"),
    ("M030","Hydroxychloroquine","Antimalarial DMARD","D012",200,0.79,280,"200-400mg daily"),
]

def gen_users(n=200):
    roles = ["Admin"]*5 + ["Analyst"]*15 + ["User"]*180
    blood = ["A+","A-","B+","B-","O+","O-","AB+","AB-"]
    conds = ["None","Hypertension","Diabetes","Asthma","Heart Disease","None","None"]
    rows = []
    import hashlib
    for i in range(n):
        uid = f"U{i+1:04d}"
        role = roles[i] if i < len(roles) else "User"
        uname = f"user{i+1}" if i >= 4 else ["admin","analyst","user","demo","mod"][i]
        pwd = "admin123" if role=="Admin" else "analyst123" if role=="Analyst" else "user123"
        phash = hashlib.sha256(pwd.encode()).hexdigest()
        rows.append({
            "user_id": uid, "username": uname,
            "password_hash": phash,
            "email": f"{uname}@healthai.com",
            "role": role,
            "age": random.randint(18, 75),
            "gender": random.choice(["Male","Female","Other"]),
            "blood_type": random.choice(blood),
            "chronic_conditions": random.choice(conds),
            "allergies": random.choice(["None","Penicillin","Sulfa","NSAIDs","Aspirin"]),
            "created_at": pd.Timestamp("2024-01-01") + pd.Timedelta(days=random.randint(0,365)),
        })
    return pd.DataFrame(rows)

def gen_diseases():
    rows = []
    for did, name, cat, sev in DISEASES:
        rows.append({"disease_id":did,"name":name,"category":cat,"severity":sev,
                     "icd_code":f"ICD-{did}","description":f"Medical condition: {name}"})
    return pd.DataFrame(rows)

def gen_medicines():
    rows = []
    for mid,name,cls,dids,dose,eff,price,dosage in MEDICINES:
        rows.append({"med_id":mid,"name":name,"category":cls,"disease_ids":dids,
                     "dosage":dosage,"effectiveness":eff,"price_inr":price,
                     "side_effects":"See package insert","contraindications":"Consult physician"})
    return pd.DataFrame(rows)

def gen_records(n=1000):
    did_map = {d[0]: d for d in DISEASES}
    rows = []
    for i in range(n):
        did, name, cat, sev = random.choice(DISEASES)
        age = random.randint(5, 85)
        rows.append({
            "record_id": f"R{i+1:05d}",
            "user_id": f"U{random.randint(1,200):04d}",
            "age": age,
            "blood_pressure_systolic": int(np.clip(np.random.normal(
                150 if did in ("D004","D005") else 120, 15), 80, 200)),
            "glucose_level": int(np.clip(np.random.normal(
                180 if did=="D004" else 100, 25), 60, 350)),
            "heart_rate": int(np.clip(np.random.normal(
                90 if did in ("D002","D008") else 75, 12), 50, 150)),
            "bmi": round(np.clip(np.random.normal(
                30 if did in ("D004","D005") else 24, 4), 15, 50), 1),
            "cholesterol": int(np.clip(np.random.normal(
                220 if did in ("D004","D005","D012") else 180, 30), 100, 350)),
            "smoking": 1 if random.random() < (0.4 if did in ("D007","D008","D005") else 0.15) else 0,
            "exercise_frequency": random.randint(0,7),
            "diagnosis": did,
            "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=random.randint(0,365)),
        })
    return pd.DataFrame(rows)

def gen_interactions(n=1500):
    rows = []
    mids = [m[0] for m in MEDICINES]
    for i in range(n):
        rows.append({
            "interaction_id": f"INT{i+1:05d}",
            "user_id": f"U{random.randint(1,200):04d}",
            "med_id": random.choice(mids),
            "rating": round(random.gauss(3.8, 0.8), 1),
            "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(days=random.randint(0,365)),
        })
    df = pd.DataFrame(rows)
    df["rating"] = df["rating"].clip(1.0, 5.0).round(1)
    return df

def gen_activity(n=3000):
    actions = ["view_medicine","search_disease","predict_disease","view_profile",
               "rate_medicine","download_report","login","logout"]
    mids = [m[0] for m in MEDICINES]
    dids = [d[0] for d in DISEASES]
    rows = []
    for i in range(n):
        action = random.choice(actions)
        ref = random.choice(mids) if "medicine" in action else random.choice(dids) if "disease" in action else ""
        rows.append({
            "log_id": f"L{i+1:05d}",
            "user_id": f"U{random.randint(1,200):04d}",
            "action": action,
            "reference_id": ref,
            "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(
                days=random.randint(0,365), hours=random.randint(0,23)),
        })
    return pd.DataFrame(rows)

def gen_reviews(n=500):
    pos = ["Excellent medication","Worked very well","Great relief","Highly recommend","No side effects"]
    neg = ["Mild side effects","Took time to work","Minor dizziness","Average results","Not very effective"]
    mids = [m[0] for m in MEDICINES]
    rows = []
    for i in range(n):
        rating = round(random.gauss(3.8, 1.0), 1)
        rating = max(1.0, min(5.0, rating))
        text = random.choice(pos) if rating >= 3.5 else random.choice(neg)
        rows.append({
            "review_id": f"REV{i+1:05d}",
            "user_id": f"U{random.randint(1,200):04d}",
            "med_id": random.choice(mids),
            "rating": rating,
            "review_text": text + f". Rating: {rating}",
            "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(days=random.randint(0,365)),
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    os.makedirs(HERE, exist_ok=True)
    gen_users(200).to_csv(HERE/"users.csv", index=False)
    gen_diseases().to_csv(HERE/"diseases.csv", index=False)
    gen_medicines().to_csv(HERE/"medicines.csv", index=False)
    gen_records(1000).to_csv(HERE/"medical_records.csv", index=False)
    gen_interactions(1500).to_csv(HERE/"interactions.csv", index=False)
    gen_activity(3000).to_csv(HERE/"activity_log.csv", index=False)
    gen_reviews(500).to_csv(HERE/"reviews.csv", index=False)
    print("✅ All datasets generated!")
