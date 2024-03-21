from configparser import ConfigParser


parser = ConfigParser()


def get_db_config(filename='v2/db/database.ini', section='postgresql'):
    parser.read(filename)
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))
    return db


def get_config(section, filename='db/local.ini'):
    parser.read(filename)
    conf = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            conf[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))
    return conf