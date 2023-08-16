INPUT_SCHEMA_PACK = {
     'api_name': {
        'type': str,
        'required': False,
    },
    'lora_key': {
        'type': str,
        'required': False,
        'default': None
    },
    'seed': {
        'type': int,
        'required': False,
        'default': None
    },
    'pack': {
        'type': object,
        'required': False,
        'default': False
    }
   
}
