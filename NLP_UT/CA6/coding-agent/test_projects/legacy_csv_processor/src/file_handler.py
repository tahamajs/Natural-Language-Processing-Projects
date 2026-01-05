import os

def load_transactions(file_path):
    """
    Reads a CSV file manually (legacy code, no pandas allowed).
    Returns a list of dictionaries. Amounts are converted to floats when possible;
    missing or invalid amounts are set to None.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    transactions = []

    with open(file_path, 'r') as f:
        lines = f.readlines()
        headers = lines[0].strip().split(',')

        for line in lines[1:]:
            values = line.strip().split(',')
            
            transaction = {
                headers[0]: values[0],
                headers[1]: values[1],
                headers[2]: values[2],
                headers[3]: values[3]
            }
            transactions.append(transaction)
            
    return transactions