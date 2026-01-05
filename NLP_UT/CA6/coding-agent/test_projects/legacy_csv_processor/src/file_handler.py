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
        lines = [ln for ln in f.readlines() if ln.strip()]
        if not lines:
            return transactions

        headers = lines[0].strip().split(',')

        for line in lines[1:]:
            values = line.strip().split(',')

            # Safely map values to headers
            record = {}
            for idx, header in enumerate(headers):
                val = values[idx] if idx < len(values) else ""
                record[header] = val

            # Convert amount to float when possible
            amount_raw = record.get('amount', '')
            try:
                amount_val = float(amount_raw) if amount_raw != "" else None
            except Exception:
                amount_val = None

            transaction = {
                'id': record.get('id', ''),
                'amount': amount_val,
                'currency': record.get('currency', ''),
                'status': record.get('status', '')
            }
            transactions.append(transaction)

    return transactions