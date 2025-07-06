import torch
from loss import ClusterLoss

class ContrastiveSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super(ContrastiveSFTTrainer, self).__init__(*args, **kwargs)
        self.cluster_loss = ClusterLoss(
            margin=kwargs['cluster_margin'], 
            metric=kwargs['cluster_metric']
        )

    def compute_contrastive_loss():
        shift_labels = labels[..., 1:].contiguous()
        mask = (shift_labels != -100)
        target = shift_labels[mask].view(mask.size(0), -1)
        feat_x = n_hidden_states[-1] #b,seq_len,h
        feat_x = feat_x[..., :-1, :].contiguous()

        b, leng, dim = feat_x.shape

        if torch.any(torch.isnan(feat_x)):
            print("nan error")
        if sim_type == "seq":
            feat_x = torch.sum(feat_x, dim=1) / leng
        elif sim_type == "masked_seq":
            feat_x = feat_x[mask]
            feat_x = feat_x[::3,:]

            # feat_x = feat_x[::3,:]+feat_x[1::3,:]
            # feat_x = feat_x / 2.0
            # mask_f = mask.type(torch.float).unsqueeze(-1).expand(-1, -1, dim)
            # feat_x = torch.mul(feat_x, mask_f)
            # feat_x = torch.sum(feat_x, dim=1) / 3.0 #torch.sum(mask, dim=1).unsqueeze(-1).expand(-1, dim), this task is 3
        elif sim_type == "token":
            pass
        elif sim_type == "linear_token":
            feat_x = feat_x[mask] #b*3,dim
            feat_x = feat_x[::3,:]
            feat_x = F.relu(self.linear_layer(feat_x.float())).half() #[b*3,dim]
            # feat_x = feat_x[::3,:]+feat_x[1::3,:]
            # feat_x = feat_x / 2.0
        elif sim_type == "linear_seq":
            feat_x = F.relu(self.linear_layer(feat_x.float())).half()
            feat_x = torch.sum(feat_x, dim=1) / leng
        else:
            raise KeyError
        
        
        index = torch.randperm(b).to(feat_x.device)
        # print("index:", index)
        feat_y = feat_x[index, :]
        # print("Feature Y Shape:", feat_y.shape)
        agreement = torch.FloatTensor([1 if x == True else 0 for x in target[:,0] == target[:, 0][index]]).to(feat_x.device)
        # print("Agreement:", agreement)
        # exit()
        loss_c = cluster_scale * torch.mean(cluster_loss(feat_x, feat_y, agreement))

    def compute_loss(self, model, inputs, return_outputs=False):
        # get label and prediction tokens   
        labels = inputs.get("labels")
        outputs = model(**inputs)
        predictions = outputs.get("logits")
    
        # decode predictions and labels
        predicted_token_ids = torch.argmax(predictions, dim=-1)
        decoded_predictions = [tokenizer.decode(p.tolist()) for p in predicted_token_ids]
        decoded_labels = [tokenizer.decode(l.tolist()) for l in labels]

        # function to output quantities to a list       
        predicted_quantities, actual_quantities = quantities(decoded_predictions, decoded_labels)
        
        predicted_tensor = torch.tensor(predicted_quantities, device=model.device)
        actual_tensor = torch.tensor(actual_quantities, device=model.device)
        predicted_tensor.requires_grad_()
        
        # Compute MSE loss
        loss_function = nn.MSELoss()
        loss = loss_function(predicted_tensor, actual_tensor)
        
        return (loss, outputs) if return_outputs else loss

     """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        mode = "eval" if self.control.should_evaluate else "train"
        (loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        if mode == "train":
            # When using padding-free, the attention_mask is not present in the inputs, instead we have cu_seq_lens_q,
            # cu_seq_lens_k, and max_length_k, max_length_q and position_ids.
            if "attention_mask" in inputs:
                num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            elif "position_ids" in inputs:
                num_tokens_in_batch = (
                    self.accelerator.gather_for_metrics(torch.tensor(inputs["position_ids"].size(1))).sum().item()
                )
            else:
                raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        # Compute token accuracy if we have labels and if the model is not using Liger (no logits)
        if "labels" in inputs and not self.args.use_liger_kernel:
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()

            # Get predictions
            predictions = shift_logits.argmax(dim=-1)

            # Create mask for non-padding tokens (assuming ignore_index is -100)
            mask = shift_labels != -100

            # Calculate accuracy only on non-padding tokens
            correct_predictions = (predictions == shift_labels) & mask
            total_tokens = mask.sum()
            correct_tokens = correct_predictions.sum()

            # Gather the correct_tokens and total_tokens across all processes
            correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
            total_tokens = self.accelerator.gather_for_metrics(total_tokens)

            # Compute the mean token accuracy and log it
            total_sum = total_tokens.sum()
            accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
            self._metrics[mode]["mean_token_accuracy"].append(accuracy)

        return (loss, outputs) if return_outputs else loss