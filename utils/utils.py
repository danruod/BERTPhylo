import torch

def get_model_output(bert, probe, batch, eval=False, return_tag=False):
    input_ids, token_type_ids, attention_mask, label = (
            batch[0],
            batch[1],
            batch[2],
            batch[3],
    )
    input_ids, token_type_ids, attention_mask, label = (
        input_ids.to(probe.device),
        token_type_ids.to(probe.device),
        attention_mask.to(probe.device),
        label.to(probe.device),
    )

    if eval:
        with torch.no_grad():
            # obtain logits and transformed for input data
            outputs = bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=True,
                output_attentions=True,
            )
            hidden_states = outputs[2]
            sequence_output = (
                hidden_states[probe.layer_num].to(probe.device).to(probe.default_dtype)
            )[:, 0, :]
            
            logits = probe(sequence_output)
    else:
        # obtain logits and transformed for input data
        outputs = bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        hidden_states = outputs[2]
        sequence_output = (
            hidden_states[probe.layer_num].to(probe.device).to(probe.default_dtype)
        )[:, 0, :]
        
        logits = probe(sequence_output)
    
    if return_tag:
        return logits, label, list(batch[4])
    
    return logits, label

