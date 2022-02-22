import torch
import math
from transformers import BertTokenizer, ViTConfig
from transformers.models.vit.modeling_vit import ViTEmbeddings
import torchvision


# https://hyugen-ai.medium.com/transformers-in-pytorch-from-scratch-for-nlp-beginners-ff3b3d922ef7
class MLPResidual(torch.nn.Module):
    def __init__(self, embedding_size: int, forward_expansion: int, dropout: float):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features=embedding_size, out_features=embedding_size*forward_expansion)
        self.linear2 = torch.nn.Linear(in_features=embedding_size*forward_expansion, out_features=embedding_size)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input_sequence: torch.Tensor):
        mapped = self.linear1(input_sequence)
        mapped = self.relu(mapped)
        mapped = self.linear2(mapped)
        mapped = self.dropout(mapped)
        return mapped


# https://hyugen-ai.medium.com/transformers-in-pytorch-from-scratch-for-nlp-beginners-ff3b3d922ef7
class Attention(torch.nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: torch.Tensor):
        attention_logits = q.matmul(k.transpose(-2, -1))  # interactions
        attention_logits = attention_logits / (q.size(-1) ** 0.5)  # normalize

        if attention_mask is not None:
            attention_logits = attention_logits.masked_fill(attention_mask[:, None, None, :] == 0, -1_000)

        attention_scores = self.softmax(attention_logits)
        attention_scores_dropout = self.dropout(attention_scores)
        out = attention_scores_dropout.matmul(v)
        return out, attention_scores


# https://hyugen-ai.medium.com/transformers-in-pytorch-from-scratch-for-nlp-beginners-ff3b3d922ef7
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embedding_size: int, num_heads: int, dropout: float = 0.2):
        """

        :param embedding_size:
        :param num_heads: embedding_size % num_heads != 0
        :param dropout:
        """
        super().__init__()
        self.num_heads = num_heads
        self.embedding_size = embedding_size
        self.head_size = int(embedding_size / num_heads)
        self.qkv_linear = torch.nn.Linear(in_features=embedding_size, out_features=embedding_size * 3)  # q, k, v
        self.attention = Attention(dropout=dropout)

    def reshape_to_heads(self, tensor: torch.Tensor):
        reshaped = tensor.reshape(tensor.size(0), -1, self.num_heads, self.head_size)
        transposed = reshaped.transpose(1, 2)
        return transposed

    def reshape_to_sequence(self, tensor: torch.Tensor):
        transposed = tensor.transpose(1, 2)
        reshaped = transposed.reshape(transposed.size(0), -1, self.embedding_size)
        return reshaped

    def forward(self, input_sequence: torch.Tensor, attention_mask: torch.Tensor):
        """

        :param input_sequence:  batch_size x seq_len x embedding_size
        :param attention_mask: batch_size x seq_len x embedding_size
        :return:
        """
        # batch_size x seq_len x 3*embedding_size
        qkv = self.qkv_linear(input_sequence)

        # each: batch_size x seq_len x embedding_size
        q = qkv[:, :, :self.embedding_size]
        k = qkv[:, :, self.embedding_size:2 * self.embedding_size]
        v = qkv[:, :, 2 * self.embedding_size:]

        # each: batch_size x num_heads x seq_len x head_size
        q, k, v = [self.reshape_to_heads(tensor) for tensor in [q, k, v]]

        # out: batch_size x num_heads x seq_len x head_size
        # attention_scores: batch_size x num_heads x seq_len x seq_len
        out, attention_scores = self.attention(q=q, k=k, v=v, attention_mask=attention_mask)

        # batch_size x seq_len x embedding_size
        out = self.reshape_to_sequence(out)
        return out, attention_scores


# https://hyugen-ai.medium.com/transformers-in-pytorch-from-scratch-for-nlp-beginners-ff3b3d922ef7
class Encoder(torch.nn.Module):
    def __init__(self, num_heads: int, embedding_size: int, forward_expansion: int, mha_dropout: float, mlp_dropout: float, encoder_dropout: float):
        super().__init__()
        self.mah = MultiHeadAttention(embedding_size=embedding_size, num_heads=num_heads, dropout=mha_dropout)
        self.mlp = MLPResidual(embedding_size=embedding_size, forward_expansion=forward_expansion, dropout=mlp_dropout)
        self.mha_norm = torch.nn.LayerNorm(embedding_size)
        self.mlp_norm = torch.nn.LayerNorm(embedding_size)
        self.encoder_dropout = torch.nn.Dropout(encoder_dropout)

    def forward(self, input_sequence: torch.Tensor, attention_mask: torch.Tensor = None):
        mha_out, attention_scores = self.mah(input_sequence=input_sequence, attention_mask=attention_mask)
        mha_out = self.mha_norm(input_sequence + mha_out)  # normed residual

        mlp_out = self.mlp(input_sequence=mha_out)
        mlp_out = self.mlp_norm(input_sequence + mlp_out)  # normed residual

        encoder_out = self.encoder_dropout(mlp_out)
        return encoder_out, attention_scores


class EmbeddingHandler(torch.nn.Module):
    def __init__(self, usage_dict: dict, dropout_dict: dict, embedding_size: int, token_seq_len: int, vit_config: ViTConfig, use_layer_norm: bool, use_segment_embedder: bool):
        super().__init__()
        self.usage_dict = usage_dict
        self.use_layer_norm = use_layer_norm
        self.just_initialized = True

        # for token and kg embedder
        tokenizer_handler = TokenizerHandler()
        vocab_size = tokenizer_handler.vocab_size
        vocab_embedder = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)

        self.use_segment_embedder = use_segment_embedder
        self.token_embedder = TokenEmbedder(vocab_embedder=vocab_embedder, embedding_size=embedding_size, seq_len=token_seq_len, dropout=dropout_dict["token"])
        self.resnet_embedder = ResNetEmbedder(embedding_size=embedding_size, dropout=dropout_dict["cnn"])
        self.vit_embedder = ViTEmbedder(vit_config=vit_config, embedding_size=embedding_size, dropout=dropout_dict["vit"])
        self.roi_embedder = ROIEmbedder(embedding_size=embedding_size, dropout=dropout_dict["roi"])
        self.kg_embedder = KGEmbedder(vocab_embedder=vocab_embedder, embedding_size=embedding_size, dropout=dropout_dict["kg"], token_seq_len=token_seq_len)
        self.text_sentiment_embedder = TextSentimentEmbedder(embedding_size=embedding_size, dropout=dropout_dict["sentiment"])
        self.segmentation_embedder = SegmentationEmbedder(embedding_size=embedding_size, num_segments=6)
        self.layer_norm = torch.nn.LayerNorm(embedding_size, elementwise_affine=False)  # if not elementwise affine, 0 mean is reached

    def forward(self, token_input_ids: torch.Tensor, token_attention_mask: torch.Tensor, image: torch.Tensor,
                detected_objects: torch.Tensor, detected_rois: torch.Tensor, object_attention_mask: torch.Tensor,
                associations: torch.Tensor, association_attention_mask: torch.Tensor,
                association_positions: torch.Tensor, vit_features: torch.Tensor, text_sentiment_features: torch.Tensor):
        device = token_input_ids.get_device()
        segmentation_ids = []
        embeddings_list = []
        attention_mask = []

        # 2 positions
        # checked
        if self.usage_dict["cnn"]:
            resnet_embeddings = self.resnet_embedder(image=image)
            embeddings_list.append(resnet_embeddings)
            attention_mask.append(torch.ones(resnet_embeddings.size(0), resnet_embeddings.size(1)))
            segmentation_ids.append(torch.ones(resnet_embeddings.size(0), resnet_embeddings.size(1))*0)

        # 32 positions
        if self.usage_dict["token"]:
            token_embeddings = self.token_embedder(token_input_ids=token_input_ids)
            embeddings_list.append(token_embeddings)
            segmentation_ids.append(torch.ones(token_embeddings.size(0), token_embeddings.size(1))*1)
            attention_mask.append(token_attention_mask)

        # 196 positions
        if self.usage_dict["vit"]:
            vit_embeddings = self.vit_embedder(vit_features)
            embeddings_list.append(vit_embeddings)
            segmentation_ids.append(torch.ones(vit_embeddings.size(0), vit_embeddings.size(1))*2)
            attention_mask.append(torch.ones(vit_embeddings.size(0), vit_embeddings.size(1)))

        # checked
        # 8 positions
        if self.usage_dict["roi"]:
            roi_embeddings = self.roi_embedder(rois=detected_rois, objects=detected_objects)
            embeddings_list.append(roi_embeddings)
            segmentation_ids.append(torch.ones(roi_embeddings.size(0), roi_embeddings.size(1))*3)
            attention_mask.append(object_attention_mask)

        # 1 position
        if self.usage_dict["sentiment"]:
            text_sentiment_embeddings = self.text_sentiment_embedder(text_sentiment_features)
            text_sentiment_embeddings = torch.unsqueeze(text_sentiment_embeddings, dim=1)
            embeddings_list.append(text_sentiment_embeddings)
            segmentation_ids.append(torch.ones(text_sentiment_embeddings.size(0), text_sentiment_embeddings.size(1))*4)
            attention_mask.append(torch.ones(text_sentiment_embeddings.size(0), text_sentiment_embeddings.size(1)))

        # 16 positions
        if self.usage_dict["kg"]:
            kg_embeddings, kg_attention_mask = self.kg_embedder(associations=associations, association_attention_mask=association_attention_mask, association_positions=association_positions)
            embeddings_list.append(kg_embeddings)
            segmentation_ids.append(torch.ones(kg_embeddings.size(0), kg_embeddings.size(1))*5)
            attention_mask.append(kg_attention_mask)

        embeddings = torch.cat(embeddings_list, dim=1).to(token_input_ids.get_device())
        # checked
        if self.use_segment_embedder:
            segmentation_ids = torch.cat(segmentation_ids, dim=1).to(token_input_ids.get_device()).long()
            segmentation_embeddings = self.segmentation_embedder(segmentation_ids=segmentation_ids)
            embeddings = embeddings + segmentation_embeddings
        # checked
        if self.use_layer_norm:
            embeddings = self.layer_norm(embeddings)  # layer norm to balance out the different features
        # checked
        attention_mask = torch.cat(attention_mask, dim=1)
        attention_mask = attention_mask.to(device)

        #if self.just_initialized:
        #    self.just_initialized = False
        #    print("transformer input is of size:", embeddings.size())
        #    print("attention mask is of size:", attention_mask.size(), "\n\n")
        return embeddings, attention_mask


# https://hyugen-ai.medium.com/transformers-in-pytorch-from-scratch-for-nlp-beginners-ff3b3d922ef7
class TokenPositionEncoder:
    def __init__(self, embedding_size: int, token_seq_len: int):
        position_encoding = torch.zeros(token_seq_len, embedding_size)
        position_encoding.requires_grad = False  # not learnable
        for position in range(token_seq_len):
            for i in range(0, embedding_size, 2):
                position_encoding[position, i] = math.sin(position / (10_000 ** ((2 * i) / embedding_size)))
                position_encoding[position, i + 1] = math.cos(position / (10_000 ** ((2 * (i + 1)) / embedding_size)))
        self.encoding = position_encoding.unsqueeze(0)


class ViTEmbedder(torch.nn.Module):
    """
    vit featuers are of size:
    batch_size x 3 x 224 x 224
    where 3 x 224 x 224 is the vit representation of one image as used in the original ViT

    vit embeddings are of size:
    batch_size x 224/16*224/16+1 x 768
    where 224/16*224/16=196 results from slicing the image into patches of size 16 x 16,
    +1 results from adding a [cls] embedding

    the cls embedding is removed, and each embedding position is mapped to the embedding_size of the network,
    so the final embeding size is:
    batch_size x 196 x embedding_size
    """
    def __init__(self, vit_config: ViTConfig, embedding_size: int, dropout: float):
        super().__init__()
        self.vit_embeddings = ViTEmbeddings(config=vit_config)
        self.vit_embeddings.dropout = torch.nn.Dropout(p=dropout)
        self.linear = torch.nn.Linear(in_features=768, out_features=embedding_size)

    def forward(self, vit_features: torch.Tensor):
        x = self.vit_embeddings(vit_features)  # module dropout is performed inside the vit embeddings
        x = x[:, 1:, :]  # discard cls
        # downscale to embedding size, no linearity so linear(embedding) falls back to just embedding
        # using just vit_embedding with a customized hidden_size might have been a better idea in retroperspective
        # but would lead to very similar results (because no nonlinearity)
        return self.linear(x)


class ResNetEmbedder(torch.nn.Module):
    def __init__(self, embedding_size: int, dropout: float):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=False)
        self.linear = torch.nn.Linear(in_features=500, out_features=embedding_size)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, image: torch.Tensor):
        x = self.resnet(image)
        positions = []
        for i in range(2):
            position = x[:, i*500:((i+1)*500)]  # here 125 and 8?
            projected_position = self.linear(position)
            unsqueezed_position = torch.unsqueeze(projected_position, dim=1)
            positions.append(unsqueezed_position)

        x = torch.cat(positions, dim=1)
        return self.dropout(x)


class TextSentimentEmbedder(torch.nn.Module):
    """
    text_sentiment_features is a tensor of size:
    batch_size x 6
    each observation is represented in this feature by the vector:
    vader_sequence_sentiment x textblob_tokenwise_sentiment_worst x textblob_tokenwise_sentiment_best x textblob_sequence_sentiment x textblob_sequence_subjectivity x rel_bad_words_amount
    """
    def __init__(self, embedding_size: int, dropout: float):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=6, out_features=embedding_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, text_sentiment_features: torch.Tensor):
        x = self.linear(text_sentiment_features)
        return self.dropout(x)


# https://hyugen-ai.medium.com/transformers-in-pytorch-from-scratch-for-nlp-beginners-ff3b3d922ef7
class TokenEmbedder(torch.nn.Module):
    """
    maps each by the tokenizer identified token and maps it to one embedding,
    adds a positional encoding to each embedding like in bert to preserve sequence information.

    dropout is performed before adding the positional encoding in order to not harming the structural information
    """
    def __init__(self, vocab_embedder: torch.nn.Embedding, embedding_size: int, seq_len: int, dropout: float):
        super().__init__()
        self.embedding = vocab_embedder
        self.position_encoding = TokenPositionEncoder(embedding_size=embedding_size, token_seq_len=seq_len)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, token_input_ids: torch.Tensor):
        embeddings = self.embedding(token_input_ids)
        embeddings = self.dropout(embeddings)  # don't use dropout on positions in this case
        position_encodings = self.position_encoding.encoding[:, :embeddings.size(1)].to(embeddings.device)
        return embeddings + position_encodings


class SegmentationEmbedder(torch.nn.Module):
    """
    adds an embedding for the segment information.
    each feature is one segment, and this embedder allows the model to distinguish between them
    """
    def __init__(self, embedding_size: int, num_segments: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=num_segments, embedding_dim=embedding_size)

    def forward(self, segmentation_ids: torch.Tensor):
        return self.embedding(segmentation_ids)


class ROIEmbedder(torch.nn.Module):
    """
    rois are the regions of interest of the detected objects.
    this module performs a simple cnn on each of the rois individually.
    the flattened convolution output is then projected to a desired embedding size.
    the output of this block is of size
    batch_size x 16 (max amount of objects by thresh) x embedding_size

    the leftover positions contain zeros (or bias when activated) only, when less than 16 objects were predicted.
    """
    def __init__(self, embedding_size: int, dropout: float):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=False)
        self.linear_roi = torch.nn.Linear(in_features=1000, out_features=embedding_size)
        self.linear_object = torch.nn.Linear(in_features=7, out_features=embedding_size)
        self.layer_norm = torch.nn.LayerNorm(embedding_size, elementwise_affine=False)  # normalise one embedding

        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, rois: torch.Tensor, objects: torch.Tensor):
        # rois: batch_size x n_rois_per_image x roi_pool_size x roi_pool_size
        convoluted_rois = []
        for roi in range(rois.size(1)):
            x = rois[:, roi, :, :]
            x = self.resnet(x)
            x = torch.unsqueeze(x, dim=1)
            convoluted_rois.append(x)
        convoluted_rois = torch.cat(convoluted_rois, dim=1)  # concatenate all convoluted rois
        roi_embeddings = self.layer_norm(self.linear_roi(convoluted_rois))
        object_embeddings = self.layer_norm(self.linear_object(objects))
        return self.dropout(roi_embeddings + object_embeddings)


class KGEmbedder(torch.nn.Module):
    """
    creates knowledge graph embeddings
    each knowledgea graph embedding is created by summing up two components:
    - an associated word to a word in the input token sequence
    - the position encoding of the original word in the input token sequence to map the association to the word
    """
    def __init__(self, vocab_embedder: torch.nn.Embedding, embedding_size: int, token_seq_len: int, dropout: float):
        super().__init__()
        self.kg = KnowledgeGraph()
        tokenizer_handler = TokenizerHandler()
        self.kg_ids = tokenizer_handler.get_kg_ids()
        self.kg_association_ids = tokenizer_handler.get_kg_associaton_ids()
        self.embedding = vocab_embedder
        self.dropout = torch.nn.Dropout(dropout)
        self.tokenizer = tokenizer_handler.tokenizer
        self.position_encoding = TokenPositionEncoder(embedding_size=embedding_size, token_seq_len=token_seq_len)  # add this on top of embedding using position

    def forward(self, associations: torch.Tensor, association_attention_mask: torch.Tensor, association_positions: torch.Tensor):
        embeddings = self.embedding(associations.long())  # creates embedding for pad token as well
        # fill every non 0 association position with its corresponding position encoding
        position_encodings = torch.zeros(embeddings.size())
        for i, obs in enumerate(association_positions):
            for j, pos in enumerate(obs):
                if association_positions[i, j] != 0:
                    position_encodings[i, j] = self.position_encoding.encoding.squeeze()[int(pos)]

        # embeddings = self.dropout(embeddings)  # not used, so effective dropout of 0 reached, no time drawback,
        # because no time for further hyperparameter optimization anyways
        embeddings = embeddings + position_encodings.to(embeddings.device)
        return embeddings, association_attention_mask


class TokenizerHandler:
    def __init__(self):
        self.kg = KnowledgeGraph()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # train the tokenizer
        for association in self.kg.associations:
            for word in association:
                self.tokenizer.add_tokens([word])
        added_vocab = self.tokenizer.get_added_vocab()
        self.vocab_size = self.tokenizer.vocab_size + len(added_vocab)

    def get_kg_ids(self):
        kg_ids = []
        for association in self.kg.associations:
            for word in association:
                kg_ids.append(self.tokenizer.encode(word)[1])
        return list(set(kg_ids))

    def get_kg_associaton_ids(self):
        kg_associaton_ids = []
        for association in self.kg.associations:
            pair = (self.tokenizer.encode(association[0])[1], self.tokenizer.encode(association[1])[1])
            kg_associaton_ids.append(pair)
        return kg_associaton_ids


class KnowledgeGraph:
    def __init__(self):

        self.associations = [
            #cluster: black
            ("black", "night"),#34275
            ("black", "crime"),#24810
            ("black", "color"),#68450
            ("monkeys", "black"),#89123
            ("monkey", "black"),#89123
            ("black", "ebola"),#40297
            ("black", "dark"),#81904
            ("black", "cotton"),#96342
            ("black", "slave"),#96342
            ("black", "plantation"),#43128
            ("black", "lazy"),#26947
            ("black", "watermelon"),#35620

            #cluster: asian
            ("asian", "bad eyes"),#04918
            ("chinese", "strange"),#58672
            ("asian", "equal"),#30267

            #cluster: racism misc
            ("supremacy", "racism"),  # 35096

            #cluster: white
            ("white", "drugs"),#96720
            ("white trash", "racism"),#29873
            ("white", "slaveholder"),#83547
            ("white", "cracker"),#81245
            ("white", "prostitution"),#42856
            ("white", "shooting"),#78532
            ("guns", "white trash"),#13902
            ("white", "snow"),#64902

            #cluster: islam
            ("muslim", "bomb"),#83152
            ("islam", "bomb"),#83152
            ("islamic", "bomb"),#83152
            ("sharia", "islam"),#91283
            ("islam", "migration"),#80217
            ("islam", "antisemitism"),#04857
            ("islam", "violence"),#41802
            ("mohammad", "islam"),#27501
            ("pig", "islam"),#27501
            ("allah", "islam"),#27501
            ("muslim", "idiot"),#27501
            ("muslim", "burden"),#58740
            ("hijab", "islam"),#67180
            ("islam", "terrorism"),#48271
            ("hijab", "diaper"),#76495
            ("islam", "sodomy"),#41387
            ("muslim", "misogyny"),#13486
            ("muslim", "homophobia"),#25103

            #cluster: christianity
            ("christian", "pedophile"),#41890
            ("church", "pedophile"),#41890

            #cluster: countries
            ("mexican", "dumb"),#39816
            ("russians", "enemy"),  # 25189
            ("russia", "enemy"),  # 25189
            ("polish", "chemical"),#34096
            ("polish", "genocide"),#29813

            #cluster: misc celebs
            ("zuckerberg", "rich"),#89056
            ("soros", "crime"),#2537



            #cluster: migration
            ("boat", "migration"),#79623
            ("black", "migration"),#03756
            ("alien", "migration"),#59328
            ("illegal", "alien"),#59328

            #cluster: animals
            ("cat", "animal"),#89073
            ("puppy", "animal"),#82943
            ("isis", "sodomy"),#48136
            ("turkey", "animal"),#56193
            ("turkey", "country"),#56193
            ("goat", "animal"),#41387
            ("goat", "sodomy"),#41387
            ("gorilla", "animal"),#20643

            #cluster: police
            ("police", "brutality"),#93542
            ("sirens", "police"),#29814
            ("police", "racism"),#92768
            ("cop", "police"),#90568
            ("cops", "police"),#90568

            #cluster: america
            ("schools", "shooting"),#48923
            ("school", "shooting"),#48923
            ("911", "terrorism"),#60329
            ("9/11", "terrorism"),#4591
            ("alabama", "incest"),#62135

            #cluster: media
            ("media", "fake news"),#56917
            ("media", "propaganda"),#76584
            ("media", "divide"),#20386

            #cluster: democrats
            ("democrat", "crime"),#31849
            ("democrats", "crime"),#31849
            ("democratic", "crime"),#31849
            ("obama", "democrat"),#41506
            ("hillary", "democrat"),#41506
            ("hillary", "turd"),#13895
            ("obama", "tranny"),#67298
            ("pelosi", "democrat"),#96380
            ("democrat", "fake news"),#43907
            ("obama", "enemy"),#25189
            ("schakowsky", "democrat"),#2389
            ("schakowsky", "islam"),#2389
            ("schakowsky", "antisemitism"),#2389
            ("obama", "illegal"),#42509
            ("hillary", "conspiracy"),#70841
            ("obama", "skidmark"),#95176

            #cluster: liberal / left
            ("liberal", "fake news"),#76584
            ("left", "triggered"),#50142

            #cluster: republicans
            ("trump", "innocent"),#68941
            ("trump", "dumb"),#09657
            ("republican", "narcissistic"),#48257
            ("republican", "white trash"),#13902
            ("republican", "racism"),#67831
            ("trump", "hooker"),#95843


            #cluster: hitler / nazis
            ("camp", "genocide"),#59124
            ("frank", "genocide"),#02946
            ("mengele", "nazi"),#19408
            ("oven", "genocide"),#61480
            ("chimney", "genocide"),#83456
            ("holocaust", "genocide"),#4263
            ("shower", "holocaust"),#84653
            ("crematorium", "holocaust"),#72850

            #cluster: disabilities
            ("down", "disability"),#35642
            ("blind", "disability"),#46315
            ("special needs", "disability"),#34897
            ("midget", "disability"),#14782
            ("vegetable", "food"),#24356
            ("vegetable","disability"),#23908

            #cluster: kitchen
            ("dishwasher", "machine"),#2459
            ("dishwasher", "kitchen"),#2459
            ("dishwasher", "woman"),#48925
            ("sandwich", "food"),#30182

            #cluster: drugs
            ("acid", "drugs"),#40693
            ("bath salts", "drugs"),#10985
            ("weed", "drug"),#18546

            #cluster:jews / israel
            ("jew", "hijacking"),#28605
            ("jews", "hijacking"),#28605
            ("jew", "creature"),#74583
            ("jewish", "jew"),#52974
            ("israel", "terrorism"),#21509
            ("jewish", "strict"),#34189
            ("holocaust", "jew"),  # 30296

            #cluster: lgbtq
            ("lgbtq", "disability"),#72514
            ("tranny", "minority"),#61204
            ("tranny", "car"),#5642
            ("lesbian", "bitter"),#85012
            ("tranny", "freak"),#19753
            ("shemale", "tranny"),#34620
            ("gay", "aids"),#91537

            #cluster: gender
            ("pronouns", "gender"),#31479
            ("tranny", "pronouns"),#31479
            ("gender", "disability"),#63871

            #cluster: women
            ("karen", "woman"),#74312
            ("karen", "annoying"),#74312
            ("abortion", "choice"),#1382
            ("sandwich", "woman"),#40918
            ("woman", "victim"),#89165
            ("girl", "annoying"),#26305
            ("girl", "woman"),#26305
            ("woman", "laundry"),#56907
            ("makeup", "degrading"),#48617
            ("granny", "woman"),#46735
            ("feminist", "ugly"),#7523
            ("feminist", "cook"),#80945
            ("feminist", "extreme"),#80297
            ("feminism", "extremism"),#80297
            ("hooker", "degrading"),#13085
            ("vacuum", "woman"),#49067
            ("hoe", "degrading"),#21753
            ("gf", "woman"),#6725
            ("ex", "degrading"),#6725

            # cluster: misogyny dataset
            ("milf", "woman"),#10461misogyny
            ("milf", "sexuality"),#10461misogyny
            ("cougar", "cat"),#4525misogyny
            ("cougar", "sexuality"),#9196misogyny
            ("housewife", "woman"),#47misogyny
            ("sexuality", "thick"),#668mis
            ("fugly", "sexuality"),#8036mis
            ("chloroform", "rape"),#7792mis
            ("toxic", "feminism"),#1361mis
            ("cheat", "sexuality"),#10618mis
            ("cheat", "game"),#6306mis
            ("cook", "girlfriend"),#8772mis
            ("cooking", "girlfriend"),#8772mis
            ("kitchen", "woman"),#4906mis
            ("women", "woman"),#4906mis

                             ]
        """
        # old
        self.associations = [("goat", "sodomy"),#hm27398
                             ("wash", "dirty"),#hm1293
                             ("muslim", "minority"),#hm50793
                             ("muslim", "terror"),#hm73204
                             ("muslim", "explode"),#hm84563
                             ("muslim", "bomb"),#hm37641
                             ("islam", "minority"),#hm50793
                             ("islam", "terror"),#hm73204
                             ("islam", "explode"),#hm84563
                             ("islam", "bomb"),#hm37641
                             ("islamic", "minority"),#hm50793
                             ("islamic", "terror"),#hm73204
                             ("islamic", "explode"),#hm84563
                             ("islamic", "bomb"),#hm37641
                             ("democrats", "politics"),#hm72941
                             ("democrat", "politics"),#hm72941
                             ("black", "crime"),#hm9315
                             ("black", "minority"),#hm53976
                             ("blacks", "crime"),#hm9315
                             ("blacks", "minority"),#53976
                             ("jew", "minority"),#20957
                             ("jews", "minority"),#hm20957
                             ("mexicans", "minority"),#hm56871
                             ("oven", "racism"),#hm61480
                             ("ex", "hate"),#hm32695
                             ("ex", "woman"),#hm32695
                             ("monkey", "minority"),#hm36724
                             ("monkeys", "minority"),#hm36724
                             ("pork", "islam"),#hm16283
                             ("quran", "islam"),#hm1682
                             ("dishwasher", "misogyny"),#hm91786
                             ("dishwasher", "machine"),#hm91754
                             ("vegetable", "minority"),#hm23908
                             ("vegetable", "plant"),#hm24356
                             ("vegetables", "plants"),#hm24356
                             ("vegetables", "minority"),#hm23908
                             ("white", "majority"),#hm93072
                             ("republicans", "politics"),#hm67810
                             ("republican", "politics"),#hm67810
                             ("obama", "politics"),#hm71364
                             ("obama", "black"),#hm71364
                             ("trump", "politics"),#hm39652
                             ("feminist", "misogyny"),#hm7523
                             ("suicide", "terrorism"),#hm34852
                             ("triggered", "hate"),#hm91724
                             ("hiroshima", "bomb"),#hm1974
                             ("nagasaki", "bomb"),#hm1974
                             ("mohammed", "islam"),#hm96415
                             ("allah", "islam"),#hm42156
                             ("karen", "hate"),#hm74312
                             ("karen", "woman"),#hm74312
                             ("tranny", "minority"),#hm67415
                             ("sandwich", "misogyny"),#hm2657
                             ("sandwich", "food"),#hm30582
                             ("sandwiches", "misogyny"),#hm2657
                             ("sandwiches", "food"),#hm30582
                             ("soros", "politician"),#hm35412  # this is not 100% correct
                             ("kitchen", "misogyny"),#hm26073
                             ("down", "minority"),#hm35642
                             ("polish", "minority"),#hm13764
                             ("hillary", "politician"),#hm3759
                             ("gorilla", "black"),#hm3845
                             ("cougar", "cat"),#mis4525
                             ("cougar", "misogyny"),#mis9196
                             ("cougars", "cat"),#mis4525
                             ("cougars", "misogyny"),#mis9196
                             ("cheat", "study"),#mis2932
                             ("cheat", "sidestep"),#mis8227
                             ("covid", "virus"),#mis3348
                             ("corona", "virus"),#mis3348
                             ("feminists", "women"),#mis10517
                             ("feminist", "woman"),#mis10517
                             ("karen", "misogyny"),#mis2040
                             ("housewife", "woman"),#mis4583
                             ("housewife", "misogyny"),#mis4583
                             ("hoe", "misogyny"),#mis5589
                             ("venomous", "hate"),#mis57
                             ("toxic", "hate"),#mis2728
                             ("offended", "minority"),#mis9729
                             ("offend", "minority"),#mis9729
                             ("gf", "woman"),#mis5861
                             ("hooker", "misogyny"),#mis7870
                             ("personality", "trait"),#mis3553
                             ("personality", "misogyny"),#mis1446
                             ("gender", "minority"),#mis9092
                             ("milf", "woman"),#mis10461
                             ("milf", "misogyny"),#mis10461
                             ("feminism", "empowerment"),#mis5077
                             ("feminism", "minority"),#mis5077
                             ("feminism", "misogyny"),#mis5077
                             ("cooking", "kitchen"),#mis4941
                             ("cooking", "misogyny"),#mis2427
                             ("cook", "kitchen"),#mis4941
                             ("cook", "misogyny"),#mis2427
                             ("period", "misogyny"),#mis11019
                             ("thick", "misogyny"),#mis668
                             ("chloroform", "misogyny"),#mis7792
                             ("alabama", "minority"),#hm62135
                             ("alabama", "state"),#hm10786
                             ("whale", "minority"),#mis3092
                             ("whale", "animal"),#mis3092
                             ("makeup", "misogyny"),#mis8093
                             ("vacuum", "misogyny"),#mis4382
                             ("vacuum", "household"),#mis1059
                             ("driving", "car"),#mis9788
                             ("driving", "misogyny"),#mis9788
                             ("drive", "car"),#mis9788
                             ("drive", "misogyny"),#mis9788
                             ("drives", "car"),#mis9788
                             ("drives", "misogyny"),#mis9788
                             ]
                        """


