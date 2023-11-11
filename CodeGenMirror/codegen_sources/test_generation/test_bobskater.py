from codegen_sources.preprocessing.obfuscation.bobskater_obfuscator import obfuscateString


f1 =\
"""def f(x): 
    foo = x
    x += 1 
    x<<foo
    return x
"""
print("testing f1")
print("function is:\n", f1)
print("obfuscated is :\n", obfuscateString(f1))


f2=\
"""def is_valid_file(file):
    if file is None:
        return False
    if isinstance(file, str):
        file = Path(file)
    else:
        assert isinstance(file, Path)
    return file.is_file() and file.stat().st_size > 0
"""
print("testing f2")
print("function is:\n", f2)
print("obfuscated is :\n", obfuscateString(f2))

f3 =\
"""def get_nlines(file_path):
    assert file_path.is_file(), file_path
    process = subprocess.run(
        f"wc -l {file_path}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.returncode == 0
    return int(process.stdout.decode().split(" ")[0])
"""
print("testing f3")
print("function is:\n", f3)
print("obfuscated is :\n", obfuscateString(f3))


f4 =\
"""def first_missing_positive ( nums ) :
    if len ( nums ) <= 0 :
        return 1
    a = 0
    for i in range ( len ( nums ) ) :
        a = nums [ i ]
        while a > 0 and a < len ( nums ) and nums [ a - 1 ] != a :
            temp = nums [ a - 1 ]
            nums [ a - 1 ] = nums [ i ]
            nums [ i ] = temp
            a = nums [ i ]
    for i in range ( len ( nums ) ) :
        if nums [ i ] != i + 1 :
            return i + 1
    return len ( nums ) + 1
"""
print("testing f4")
print("function is:\n", f4)
print("obfuscated is :\n", obfuscateString(f4))
