import sys
from src.file_handler import load_transactions
from src.analyzer import calculate_total_revenue, get_average_transaction

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <csv_file>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    print(f"Processing {file_path}...")
    
    try:
        data = load_transactions(file_path)
        print(f"Loaded {len(data)} transactions.")
        
        total = calculate_total_revenue(data)
        print(f"Total Revenue: ${total:.2f}")
        
        avg = get_average_transaction(data)
        print(f"Average: ${avg:.2f}")
        
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
