import { Module } from '@nestjs/common';
import { RunsController } from './runs.controller';
import { RunsService } from './runs.service';
import { Phase1EngineClient } from './phase1-engine.client';
import {
  Phase1ResultPersistence,
} from './phase1-result.persistence';

@Module({
  controllers: [RunsController],
  providers: [
    RunsService,
    Phase1EngineClient,
    Phase1ResultPersistence,
  ],
})
export class RunsModule {}
