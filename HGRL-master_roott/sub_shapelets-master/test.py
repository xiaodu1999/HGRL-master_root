dataset = {
    'AtrialFibrillation': {
        'prefix': 'This subject age is',
        'suffix': 'years old',
        'sublabel': {
            0: '20',
            1: '30',
            2: '40',
            3: '50'
        }
    }
}

# 访问 sublabel
sublabel = dataset['AtrialFibrillation']['sublabel']
print(sublabel)  # 这应该不会引发错误
