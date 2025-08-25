import os

dut_names_dict = {
    "0xf820357251287ce1": "BE-1",
    "0x07540ba7e21004bf": "BE-2",
    "0x679f80f90cea91dd": "BE-3",
    "0x77db265172dec07f": "BE-4",
    "0xd0213d8230e21d84": "BE-5",
    "0x862b09ac4bbb4604": "BE-6",
    "0xb14c93f83f6fa826": "BE-7",
    "0x9c15f93d401b4598": "BE-8",
    "0x9c15f93d401b4598": "BE-8",
    "0x5c84a4c09ecf87d8": "BE-9",
}

for root, dirs, files in os.walk('.'):
    for filename in files:
        new_filename = filename
        for key, value in dut_names_dict.items():
            if key and key in new_filename:
                new_filename = new_filename.replace(key, value)
        if new_filename != filename:
            src = os.path.join(root, filename)
            dst = os.path.join(root, new_filename)
            os.rename(src, dst)
            print(f"Renamed: {src} -> {dst}")