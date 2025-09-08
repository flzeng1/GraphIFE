def parameter_visualization(args):
    args_dict = vars(args)
    max_key_len = max(len(key) for key in args_dict.keys())
    lines = []
    for key, value in args_dict.items():
        print(f"{key.ljust(max_key_len)} : {value}")
        lines.append(f"{key.ljust(max_key_len)} : {value}")
    return "\n".join(lines)