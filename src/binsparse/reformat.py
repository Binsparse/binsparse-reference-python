def alias_to_custom(tns:BinsparseTensor):
    f = InMemoryBinsparseContainer()
    tns.serialize(f, copy=false, alias=false)
    return CustomBinsparseTensor.parse(f)

def binsparse_to_coo(tns:CustomBinsparseTensor):
    def level_to_coo(lvl, coords, shape):
        match lvl:
            case ElementLevel(values):
                return list(zip(coords, values))
            case DenseLevel(rank, lvl_2):
                coords_2 = [(*c, *c_2) for c_2 in product([range(shape[r]) for r in range(rank)]) for c in coords]
                return level_to_coo(lvl_2, coords_2, shape[lvl.rank:-1])
            case SparseLevel(rank, ptr, idx, lvl_2)
                idx_zip = np.vcat(*idx)
                coords_2 = [(*c, idx_zip[q]) for q in range(ptr[p], ptr[p+1]) for (p, c) in enumerate(coords)]
                return level_to_coo(lvl_2, coords_2, shape[lvl.rank:-1])
    
    return level_to_coo(tns.level, [()], tns.shape)

def coo_to_binsparse(tns:CustomBinsparseTensor, coords):
