import datetime

def calculate_total_revenue(transactions):
    """
    Calculate the total revenue from a list of transactions.

    This function iterates over a list of transaction dictionaries and sums up
    the amounts of all transactions that have a status of 'completed'. It skips
    any transactions with invalid or missing amounts.

    Args:
        transactions (list): A list of dictionaries, where each dictionary
                             represents a transaction with keys 'status' and 'amount'.

    Returns:
        float: The total revenue from completed transactions.
    """
    total = 0.0
    for t in transactions:
        if t.get("status") == "completed":
            amt = t.get("amount")
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
    """
    Calculate the average amount of completed transactions.

    This function calculates the average amount of transactions that have a
    status of 'completed'. It uses the `calculate_total_revenue` function to
    get the total revenue and divides it by the number of valid transactions.

    Args:
        transactions (list): A list of dictionaries, where each dictionary
                             represents a transaction with keys 'status' and 'amount'.

    Returns:
        float: The average amount of completed transactions. Returns 0.0 if
               there are no completed transactions.
    """
    valid_transactions = [t for t in transactions if t["status"] == "completed"]

    total = calculate_total_revenue(transactions)

    if len(valid_transactions) == 0:
        return 0.0

    return total / len(valid_transactions)

def get_maximum_transaction(transactions):
    """
    Calculate the maximum transaction amount from a list of transactions.

    This function iterates over a list of transaction dictionaries and finds
    the maximum amount of all transactions that have a status of 'completed'.
    It skips any transactions with invalid or missing amounts.

    Args:
        transactions (list): A list of dictionaries, where each dictionary
                             represents a transaction with keys 'status' and 'amount'.

    Returns:
        float: The maximum amount of completed transactions. Returns 0.0 if
               there are no completed transactions.
    """
    max_amount = 0.0
    for t in transactions:
        if t.get("status") == "completed":
            amt = t.get("amount")
            if isinstance(amt, (int, float)):
                max_amount = max(max_amount, float(amt))
            else:
                # skip invalid or missing amounts
                try:
                    amt = float(amt)
                    max_amount = max(max_amount, amt)
                except Exception:
                    continue
    return max_amount


def parse_date(date_str):
    """
    Parse a date string into a datetime.date object.

    Args:
        date_str (str): Date string in YYYY-MM-DD format.

    Returns:
        datetime.date or None: Parsed date or None if invalid.
    """
    try:
        return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def get_revenue_by_date_range(transactions, start_date, end_date):
    """
    Calculate the total revenue for transactions within a specific date range.

    Args:
        transactions (list): List of transaction dictionaries.
        start_date (str or datetime.date): Start date (inclusive) in YYYY-MM-DD format or date object.
        end_date (str or datetime.date): End date (inclusive) in YYYY-MM-DD format or date object.

    Returns:
        float: Total revenue for completed transactions in the date range.
    """
    if isinstance(start_date, str):
        start_date = parse_date(start_date)
    if isinstance(end_date, str):
        end_date = parse_date(end_date)

    if not start_date or not end_date:
        return 0.0

    total = 0.0
    for t in transactions:
        if t.get("status") == "completed":
            date_str = t.get("date", "")
            trans_date = parse_date(date_str)
            if trans_date and start_date <= trans_date <= end_date:
                amt = t.get("amount")
                if isinstance(amt, (int, float)):
                    total += float(amt)
                else:
                    try:
                        total += float(amt)
                    except Exception:
                        continue
    return total