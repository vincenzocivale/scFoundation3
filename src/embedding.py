def infer_embeddings_batch(
    npz_path,
    gene_list_path,
    checkpoint_path,
    save_dir,
    cell_id_list=None,
    batch_size=64,
    pre_normalized="F",
    tgthighres="t4",
    pool_type="all"
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Caricamento modello...")
    model, config = load_model_frommmf(checkpoint_path, key="cell")
    model.eval().to(device)

    print("Caricamento dati...")
    gene_list = pd.read_csv(gene_list_path, sep="\t")["gene_name"].tolist()
    X = scipy.sparse.load_npz(npz_path).toarray()
    df = pd.DataFrame(X)
    df.index = [f"cell_{i}" for i in range(df.shape[0])] if cell_id_list is None else cell_id_list

    missing = list(set(gene_list) - set(df.columns))
    if missing:
        df_missing = pd.DataFrame(0, index=df.index, columns=missing)
        df = pd.concat([df, df_missing], axis=1)
    df = df[gene_list]

    # Se è stata passata una lista, filtriamo
    if cell_id_list is not None:
        df = df.loc[cell_id_list]

    print(f"Totale cellule da processare: {len(df)}")

    for i in tqdm(range(0, len(df), batch_size), desc="Batch inference"):
        batch_df = df.iloc[i:i+batch_size]
        batch_exprs = []
        add_meta_all = []

        for _, row in batch_df.iterrows():
            if pre_normalized == "F":
                normed = np.log1p(row / row.sum() * 1e4)
            elif pre_normalized == "T":
                normed = row
            elif pre_normalized == "A":
                normed = row[:-1]
            else:
                raise ValueError("Invalid pre_normalized")

            total_count = row.sum() if pre_normalized != "A" else row[-1]

            if tgthighres[0] == 't':
                add_meta = [float(tgthighres[1:]), np.log10(total_count)]
            elif tgthighres[0] == 'f':
                add_meta = [np.log10(total_count * float(tgthighres[1:])), np.log10(total_count)]
            elif tgthighres[0] == 'a':
                add_meta = [np.log10(total_count) + float(tgthighres[1:]), np.log10(total_count)]
            else:
                raise ValueError("tgthighres malformato")

            batch_exprs.append(np.concatenate([normed, add_meta]))
            add_meta_all.append(add_meta)

        batch_tensor = torch.tensor(batch_exprs).to(device)
        gene_ids = torch.arange(batch_tensor.shape[1], device=device).unsqueeze(0).repeat(batch_tensor.shape[0], 1)

        value_mask = batch_tensor > 0
        x, x_pad = gatherData(batch_tensor, value_mask, config['pad_token_id'])
        pos_ids, _ = gatherData(gene_ids, value_mask, config['pad_token_id'])

        x_embed = model.token_emb(x.unsqueeze(2).float(), output_weight=0) + model.pos_emb(pos_ids)
        encoded = model.encoder(x_embed, x_pad)

        emb1 = encoded[:, -1, :]
        emb2 = encoded[:, -2, :]
        emb3, _ = torch.max(encoded[:, :-2, :], dim=1)
        emb4 = torch.mean(encoded[:, :-2, :], dim=1)

        if pool_type == "all":
            final_emb = torch.cat([emb1, emb2, emb3, emb4], dim=1)
        elif pool_type == "max":
            final_emb, _ = torch.max(encoded, dim=1)
        else:
            raise ValueError("pool_type deve essere 'all' o 'max'")

        for j, cell_id in enumerate(batch_df.index):
            np.save(os.path.join(save_dir, f"{cell_id}.npy"), final_emb[j].detach().cpu().numpy())
