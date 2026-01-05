def calculate_total_revenue(transactions):
    """Calculates sum of all completed transactions."""
    total = 0.0
    for t in transactions:
        if t['status'] == 'completed':
            total += t['amount'] 
    return total

def get_average_transaction(transactions):
    """Calculates average transaction amount."""
    valid_transactions = [t for t in transactions if t['status'] == 'completed']
    
    total = calculate_total_revenue(transactions)
    
    return total / len(valid_transactions)