program_string = '''global position = [0,0,0,0,0,0]

def convert_port_to_val(port):
    val = read_port_register(port)
    if val > 32768:
        val = val - 65536
    end
    return val
end

def update_position():
    position[0] = convert_port_to_val(128)*0.01
    position[1] = convert_port_to_val(129)*0.01
    position[2] = convert_port_to_val(130)*0.01
    position[3] = convert_port_to_val(131)*0.01
    position[4] = convert_port_to_val(132)*0.01
    position[5] = convert_port_to_val(133)*0.01
end

while(True):
    update_position()
    allj = get_inverse_kin(p[position[0],position[1],position[2],position[3],position[4],position[5]])
    servoj(allj,0.5,0.5,0.5)
end'''

print(program_string)