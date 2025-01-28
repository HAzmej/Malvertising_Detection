def featureengineering(test):
    import pandas as pd
    # import numpy as np
    
    dataset=pd.DataFrame()
    #Length oui
    parent=test["parent_url"]
    dataset["parent_url_len"]=parent.apply(len)


    import tldextract
    
    #TLDlen oui
    dt=test['parent_url'].apply(lambda url: tldextract.extract(url).suffix)

    dataset["TLD_len"]=dt.apply(len)

    #NoOfSubDomain oui
    from urllib.parse import urlparse

    def count_subdomains(url):
        """
        Calcule le nombre de sous-domaines d'une URL donnée.
        """
        parsed_url = urlparse(url)
        domain = parsed_url.netloc


        domain_parts = domain.split('.')


        if len(domain_parts) > 2:

            subdomain_count = len(domain_parts) - 2
        else:

            subdomain_count = 0

        return subdomain_count
    dataset["NoOfSubDomain"]=test["parent_url"].apply(count_subdomains)

    #HasObfuscation exemple: ...malicious-site.com/?
    import re
    # from urllib.parse import unquote

    def has_obfuscation(url):
        """
        Détecte si une URL contient des signes d'obfuscation.
        """
        # Vérification de l'encodage URL (présence de caractères encodés)
        if '%' in url:
            return True

        # Vérification de la présence de chiffres dans les mots (ex. l0g1n au lieu de login)
        words = re.findall(r'\w+', url)  # Extraire tous les mots (composés de lettres et chiffres)
        for word in words:
            if re.search(r'\d', word):  # Si le mot contient des chiffres
                return True

        # Vérification de la présence de caractères spéciaux typiques
        if re.search(r'[!@#$%^&*(),.?":{}|<>]', url):
            return True

        # Si aucune obfuscation détectée
        return False
    dataset["HasObfuscation"]=test["parent_url"].apply(has_obfuscation)

    #ObFRatio oui
    import re
    from urllib.parse import unquote

    def obfuscation_ratio(url):
        """
        Calcule le ratio d'obfuscation d'une URL en fonction de la présence d'encodages,
        de chiffres dans les mots, de caractères spéciaux et d'autres signes d'obfuscation.
        """
        # Calculer la longueur totale de l'URL (en caractères)
        total_length = len(url)

        # Calculer la longueur des parties obfusquées
        obfuscated_length = 0

        # Vérification de l'encodage URL
        encoded_chars = len(re.findall(r'%[0-9A-Fa-f]{2}', url))  # Caractères encodés comme %20
        obfuscated_length += encoded_chars

        # Vérification de la présence de chiffres dans les mots
        words = re.findall(r'\w+', url)  # Extraire tous les mots (composés de lettres et chiffres)
        for word in words:
            if re.search(r'\d', word):  # Si le mot contient des chiffres
                obfuscated_length += len(word)

        # Vérification de la présence de caractères spéciaux
        special_chars = len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', url))
        obfuscated_length += special_chars

        # Calculer le ratio d'obfuscation (obfuscated_length / total_length)
        if total_length > 0:
            return obfuscated_length / total_length
        else:
            return 0
    dataset["ObfRatio"]=test["parent_url"].apply(obfuscation_ratio)
    

    #NoOfLettersInURL oui
    import re

    def no_of_letters_in_url(url):
        """
        Calcule le nombre de lettres (alphabétiques) dans une URL.
        """
        letters = re.findall(r'[a-zA-Z]', url)


        return len(letters)
    dataset["NoOfLettersInURL"]=test["parent_url"].apply(no_of_letters_in_url)

    #letter_ratio_in_url oui
    dataset['letter_ratio_in_url'] = dataset['NoOfLettersInURL'] / dataset['parent_url_len']

    #NoOfDigitsInURL oui
    def no_of_digits_in_url(url):
        """
        Calcule le nombre de chiffres dans une URL.
        """

        digits = re.findall(r'\d', url)


        return len(digits)
    
    dataset["NoOfDigitsInURL"]=test["parent_url"].apply(no_of_digits_in_url)

    #Digits_ratio_in_url oui
    dataset['Digits_ratio_in_url'] = dataset['NoOfDigitsInURL'] / dataset['parent_url_len']

    #NoOfEqualsInURL oui
    def no_of_equals_in_url(url):
        """
        Calcule le nombre de signes égaux (=) dans une URL.
        """
        return url.count('=')
    dataset["NoOfEqualsInURL"]=test["parent_url"].apply(no_of_equals_in_url)

    #NoOfQMarkInURL oui
    def no_of_qmark_in_url(url):
        """
        Calcule le nombre de signes égaux (=) dans une URL.
        """
        return url.count('?')
    dataset["NoOfQMarkInURL"]=test["parent_url"].apply(no_of_qmark_in_url)

    #NoOfAmpersandInURL &
    def no_of_ands_in_url(url):
        """
        Calcule le nombre de signes égaux (=) dans une URL.
        """
        return url.count('&')
    dataset["NoOfAmpersandInURL"]=test["parent_url"].apply(no_of_ands_in_url)

    #NoOfOtherSpecialCharsInURL
    def no_of_other_special_chars_in_url(url):
        """
        Calcule le nombre de caractères spéciaux autres que les lettres, chiffres, = et & dans une URL.
        """
        special_chars = re.findall(r'[^a-zA-Z0-9=&/?:#-]', url)
        return len(special_chars)
    dataset["NoOfOtherSpecialCharsInURL"]=test["parent_url"].apply(no_of_other_special_chars_in_url)

    #SpacialCharRatioInURL
    dataset["SpacialCharRatioInURL"]=dataset["NoOfOtherSpecialCharsInURL"]/dataset["parent_url_len"]

    #IsHTTPS 
    # def is_https(url):
    #     """
    #     Vérifie si une URL utilise le protocole HTTPS.
    #     """
    #     return url.lower().startswith('https://')
    # dataset["IsHTTPS"]=dataset["parent_url"].apply(is_https)

    #Save as newDataset.csv
    print(dataset)
    return dataset
