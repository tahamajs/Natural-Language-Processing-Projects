def calculate_total_revenue(transactions):
    """Calculates sum of all completed transactions."""
    total = 0.0
    for t in transactions:
        if t.get('status') == 'completed':
            amt = t.get('amount')
            if isinstance(amt, (int, float)):
                total += float(amt)
            else:
                # skip invalid or missing amounts
                try:
                    total += float(amt)
                except Exception:
                    continue
    return total

def get_average_transaction(transactions):
    """Calculates average transaction amount."""
    valid_transactions = [t for t in transactions if t['status'] == 'completed']
    
    total = calculate_total_revenue(transactions)

    if len(valid_transactions) == 0:
        return 0.0

    return total / len(valid_transactions)