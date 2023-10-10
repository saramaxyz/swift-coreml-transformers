//
//  GPT2.swift
//  CoreMLGPT2
//
//  Created by Julien Chaumond on 19/07/2019.
//  Copyright © 2019 Hugging Face. All rights reserved.
//

import Foundation
import CoreML
import AeroEdge

class GPT2 {
    
    enum DecodingStrategy {
        /// At each time step, we select the most likely next token
        case greedy
        /// Sample only from the top-k most-probable tokens (k is a hyper-parameter).
        case topK(Int)
        /// Sample from the top tokens with a cumulative probability just above a threshold (nucleus/top-p).
        case topP(Double)
    }
    
//    private let model = gpt2_64_12()
    public let tokenizer = GPT2Tokenizer()
    public let seqLen = 64
    private let strategy: DecodingStrategy
    private var model: MLModel?

    init(aeroEdge: AeroEdge, strategy: DecodingStrategy = .greedy ) {
        self.strategy = strategy
        Task {
            print("Downloading AeroEdge model...")
            await aeroEdge.getModel(modelName: "gpt2-64-12", bundledModelURL: nil, progress: { progress in
                print("Download Progress: \(progress)")
            }, completion: { [weak self] result, isFinal in
                switch result {
                case .success(let model):
                    print(model.description)
                    self?.model = model
                case .failure(let error):
                    // Handle the error
                    print(error.localizedDescription)
                }
            }
            )
        }
    }
    
    /// Main prediction loop:
    /// Predict next token from array of previous tokens.
    /// - featurization
    /// - model inference
    /// - Decoding according to the model's `strategy`
    func predict(tokens: [Int]) -> Int {
        let maxTokens = (tokens.count > seqLen)
            ? Array(tokens[..<seqLen])
            : tokens
        
        /// Pad input_ids on the right, up to `seqLen`:
        let input_ids = MLMultiArray.from(
            maxTokens + Array(repeating: 0, count: seqLen - maxTokens.count)
        )
        let position_ids = MLMultiArray.from(
            Array(0..<seqLen)
        )
        
        let output = try! model!.prediction(from: MLDictionaryFeatureProvider(dictionary: ["input_ids": input_ids, "position_ids": position_ids]))
        
        let outputLogits = MLMultiArray.slice(
            output.featureValue(for: "output_logits")!.multiArrayValue!,
            indexing: [.select(0), .select(maxTokens.count - 1), .slice, .select(0), .select(0)]
        )
        
        switch strategy {
        case .greedy:
            let nextToken = Math.argmax(outputLogits)
            return nextToken.0
        case .topK(let k):
            let logits = MLMultiArray.toDoubleArray(outputLogits)
            let topk = Math.topK(arr: logits, k: k)
            let sampleIndex = Math.sample(indexes: topk.indexes, probs: topk.probs)
            return sampleIndex
        case .topP(_):
            fatalError("topP is not implemented yet")
        }
    }
    
    
    /// Main generation loop.
    ///
    /// Will generate next `nTokens` (defaults to 10).
    /// Calls an incremental `callback` for each new token, then returns the generated string at the end.
    ///
    func generate(text: String, nTokens: Int = 10, callback: ((String, Double) -> Void)?) -> String {
        var tokens = tokenizer.encode(text: text)
        var newTokens: [Int] = []
        for i in 0..<nTokens {
            let (nextToken, time) = Utils.time {
                return predict(tokens: tokens)
            }
            
            tokens.append(nextToken)
            newTokens.append(nextToken)
            print("🦄 <\(time)s>", i, nextToken, tokens.count)
            callback?(
                tokenizer.decode(tokens: newTokens), time
            )
        }
        return tokenizer.decode(tokens: newTokens)
    }
}
