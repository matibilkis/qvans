def give_proper_indices(original_db):
    first_symbol = original_db["symbol"].dropna().iloc[0]
    original_ind = original_db.index[0]
    number = first_symbol.replace("th_","")
    return int(number), original_ind
