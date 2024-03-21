from cryptography.fernet import Fernet
import io
import msoffcrypto
from v2.config.app_config import get_config


def save_key(key):
    with open("encryption.key", "wb") as key_file:
        key_file.write(key)


def load_key():
    return open("encryption.key", "rb").read()


def generate_key():
    key = Fernet.generate_key()
    save_key(key)


def encrypt_text(text: str):
    """
    Encrypts a message
    """
    key = load_key()
    encoded_message = text.encode()
    fernet = Fernet(key)
    encrypted_message = fernet.encrypt(encoded_message)
    encrypted_message_str = encrypted_message.decode('utf-8')
    return encrypted_message_str


def decrypt_text(encrypted_message_str: str):
    """
    Decrypts an encrypted message
    """
    encrypted_message = encrypted_message_str.encode()
    key = load_key()
    f = Fernet(key)
    decrypted_message = f.decrypt(encrypted_message)
    decrypted_message_str = decrypted_message.decode('utf-8')
    return decrypted_message_str


def decrypt_file(path_to_file, employee_type: str = 'perm'):
    config = get_config(section='secrets')
    passwd = config.get(f'salary_{employee_type}')
    decrypted_workbook = io.BytesIO()
    with open(path_to_file, 'rb') as file:
        office_file = msoffcrypto.OfficeFile(file)
        office_file.load_key(password=passwd)
        office_file.decrypt(decrypted_workbook)
    return decrypted_workbook
