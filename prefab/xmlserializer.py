HEADER = "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
METADATA = "xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xmlns:xsd=\"http://www.w3.org/2001/XMLSchema\""
TABSPACE = 2

def isempty(data):
    if data is None:
        return True
    if type(data) == dict:
        for v in data.values():
            if v != None and v != []:
                return False
        return True
    else:
        return not bool(data)

def serialize(data, object_name, tablevel=0):
    indent = " " * tablevel * TABSPACE
    if isempty(data):
        return indent + "<%s />\n" % object_name
    if type(data) == list:
        string = ""
        for item in data:
            string += serialize(item, object_name, tablevel=tablevel+1)
        string += indent
    else:
        string = indent + "<%s>" % object_name
        if type(data) == dict:
            string += "\n"
            for k, v in data.items():
                string += serialize(v, k, tablevel=tablevel+1)
            string += indent
        else:
            string += str(data)
        string += "</%s>\n" % object_name
    return string

def write(data, root_name, path):
    with open(path, 'w+', -1, 'utf-8') as f:
        f.write(HEADER)
        f.write("<%s %s>\n" % (root_name, METADATA))
        for k, v in data.items():
            xmlstring = serialize(v, k, tablevel=1)
            f.write(xmlstring)
        f.write("</%s>\n" % root_name)
