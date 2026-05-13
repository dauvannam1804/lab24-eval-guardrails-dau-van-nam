import pandas as pd
import os

def check_testset():
    file_path = "phase-a/testset_v1.csv"
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File {file_path} not found!")
        return

    # 1. Load data
    df = pd.read_csv(file_path)
    num_rows = len(df)
    
    print("="*50)
    print("📊 TEST SET VERIFICATION (TASK A.1)")
    print("="*50)

    # 2. Check row count
    if num_rows >= 50:
        print(f"✅ Row count: {num_rows} (PASS: >= 50)")
    else:
        print(f"❌ Row count: {num_rows} (FAIL: Expected at least 50)")

    # 3. Check distribution
    print("\n🔍 Distribution by Synthesizer Name:")
    if 'synthesizer_name' in df.columns:
        dist = df['synthesizer_name'].value_counts()
        dist_pct = df['synthesizer_name'].value_counts(normalize=True) * 100
        
        for name, count in dist.items():
            pct = dist_pct[name]
            print(f"  - {name}: {count} rows ({pct:.1f}%)")
    else:
        print("❌ Error: Column 'synthesizer_name' not found in CSV!")

    # 4. Check for null values
    null_counts = df.isnull().sum()
    if null_counts.sum() == 0:
        print("\n✅ Missing values: None (PASS)")
    else:
        print(f"\n⚠️ Warning: Found missing values:\n{null_counts[null_counts > 0]}")

    print("="*50)

if __name__ == "__main__":
    check_testset()
