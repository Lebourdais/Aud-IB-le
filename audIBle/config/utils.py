def merge_configs(common_params, specific_config):
    """
    Fusionne un dictionnaire de paramètres communs avec une configuration spécifique.
    
    La fusion est récursive pour les dictionnaires imbriqués, permettant de ne modifier que
    les paramètres spécifiés dans la configuration spécifique.
    
    Args:
        common_params (dict): Dictionnaire des paramètres communs
        specific_config (dict): Dictionnaire de la configuration spécifique à fusionner
        
    Returns:
        dict: Un nouveau dictionnaire contenant les paramètres fusionnés
    """
    # Créer une copie profonde des paramètres communs pour éviter de les modifier
    import copy
    merged = copy.deepcopy(common_params)
    
    # Parcourir le dictionnaire de configuration spécifique
    for key, value in specific_config.items():
        # Si la valeur est un dictionnaire et que la clé existe déjà dans les paramètres communs
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            # Récursion pour fusionner les sous-dictionnaires
            merged[key] = merge_configs(merged[key], value)
        else:
            # Sinon, remplacer/ajouter la valeur
            merged[key] = copy.deepcopy(value)
    
    return merged